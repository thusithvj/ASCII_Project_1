"""
GLOBAL MULTIVARIATE LSTM (3 outputs) + 27 PLOTS (per center per meal)
--------------------------------------------------------------------
Dataset (your CSV) columns used ONLY:
  Targets:
    - BreakFast, Lunch, Dinner

  Features:
    - IsHoliday
    - DayOfWeek_0..DayOfWeek_6  (one-hot)
    - BreakFast_lag_1d, BreakFast_lag_1w, BreakFast_lag_1m
    - Lunch_lag_1d, Lunch_lag_1w, Lunch_lag_1m
    - Dinner_lag_1d, Dinner_lag_1w, Dinner_lag_1m

Model:
  Input:  (LOOKBACK, num_features)
  Output: (3,) -> [BreakFast, Lunch, Dinner] for next day

Plots:
  Saves 27 separate plots:
    ./plots_per_center_per_meal/rc_<id>_<meal>_test_pred_vs_actual.png
"""

# ==============================
# 0) Imports & Config
# ==============================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

CSV_PATH = "C:\\Users\\thusi\\Downloads\\Ascii\\global_breakfast_dataset.csv"
MODEL_DIR = "./models_lstm_allmeals"
PLOTS_DIR = "./plots_per_center_per_meal"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

DATE_COL = "Date"
GROUP_COL = "RevenueCenterID"

TARGET_COLS = ["BreakFast", "Lunch", "Dinner"]

# Feature columns (ONLY what you requested)
DOW_COLS = [f"DayOfWeek_{i}" for i in range(7)]
LAG_COLS = [
    "BreakFast_lag_1d", "BreakFast_lag_1w", "BreakFast_lag_1m",
    "Lunch_lag_1d",     "Lunch_lag_1w",     "Lunch_lag_1m",
    "Dinner_lag_1d",    "Dinner_lag_1w",    "Dinner_lag_1m",
]
FEATURE_COLS = ["IsHoliday"] +["Holiday_Yesterday"] + ["Holiday_LastWeek"] + DOW_COLS + LAG_COLS

LOOKBACK = 28
HORIZON = 1

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10

BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-3


# ==============================
# 1) Load & Clean
# ==============================
def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if DATE_COL not in df.columns or GROUP_COL not in df.columns:
        raise ValueError(f"CSV must contain {DATE_COL} and {GROUP_COL}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)

    # Sanity checks for required cols
    required = [DATE_COL, GROUP_COL] + TARGET_COLS + FEATURE_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Keep only what we want (ensures rolling/EMA etc. are not used)
    df = df[[DATE_COL, GROUP_COL] + TARGET_COLS + FEATURE_COLS].copy()

    # Drop any rows that have missing values in targets/features
    df = df.dropna(subset=TARGET_COLS + FEATURE_COLS).reset_index(drop=True)

    return df


# ==============================
# 2) Split helper + sequence builder
# ==============================
def split_indices_per_group(n: int, train_ratio: float, val_ratio: float):
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return train_end, val_end


def make_sequences(X_2d: np.ndarray, y_2d: np.ndarray, dates: np.ndarray, lookback: int, horizon: int):
    """
    X_2d: (T, F)
    y_2d: (T, 3)
    returns:
      X: (N, lookback, F)
      y: (N, 3)
      y_dates: (N,)  label dates
    """
    X, y, y_dates = [], [], []
    T = len(X_2d)
    last_start = T - lookback - horizon + 1
    if last_start <= 0:
        return (
            np.empty((0, lookback, X_2d.shape[1]), dtype=np.float32),
            np.empty((0, y_2d.shape[1]), dtype=np.float32),
            np.empty((0,), dtype="datetime64[ns]"),
        )

    for start in range(last_start):
        end = start + lookback
        label_idx = end + horizon - 1
        X.append(X_2d[start:end, :])
        y.append(y_2d[label_idx, :])
        y_dates.append(dates[label_idx])

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(y_dates, dtype="datetime64[ns]"),
    )


# ==============================
# 3) Prepare global train/val/test (stacked across centers)
# ==============================
def prepare_global_datasets(df: pd.DataFrame):
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []

    centers = df[GROUP_COL].unique()

    for cid in centers:
        g = df[df[GROUP_COL] == cid].sort_values(DATE_COL).reset_index(drop=True)
        if len(g) < (LOOKBACK + HORIZON + 10):
            continue

        n = len(g)
        train_end, val_end = split_indices_per_group(n, TRAIN_RATIO, VAL_RATIO)

        # --- Targets: scale per center (fit on TRAIN only) ---
        y_raw = g[TARGET_COLS].to_numpy(dtype=np.float32)  # (T,3)
        y_scaler = StandardScaler()
        y_scaler.fit(y_raw[:train_end])  # no leakage
        y_scaled = y_scaler.transform(y_raw)

        # --- Features (already numeric) ---
        X_feat = g[FEATURE_COLS].to_numpy(dtype=np.float32)  # (T,F)
        dates = g[DATE_COL].to_numpy(dtype="datetime64[ns]")

        # windows
        X_all, y_all, _ = make_sequences(X_feat, y_scaled, dates, LOOKBACK, HORIZON)
        label_indices = np.arange(len(y_all)) + LOOKBACK + HORIZON - 1

        train_mask = label_indices < train_end
        val_mask = (label_indices >= train_end) & (label_indices < val_end)
        test_mask = label_indices >= val_end

        if train_mask.any():
            X_train_list.append(X_all[train_mask])
            y_train_list.append(y_all[train_mask])
        if val_mask.any():
            X_val_list.append(X_all[val_mask])
            y_val_list.append(y_all[val_mask])
        if test_mask.any():
            X_test_list.append(X_all[test_mask])
            y_test_list.append(y_all[test_mask])

    if not X_train_list:
        raise RuntimeError("No training sequences created. Check data lengths / lookback / NaNs.")

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ==============================
# 4) Multivariate LSTM model (3 outputs)
# ==============================
def build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(LOOKBACK, input_dim)),
        layers.LSTM(128),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dense(3)  # BreakFast, Lunch, Dinner
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.Huber(delta=1.0),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model


# ==============================
# 5) Train + evaluate
# ==============================
def train_model(model: keras.Model, train_data, val_data):
    X_train, y_train = train_data
    X_val, y_val = val_data

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "best_lstm.keras"),
                                        monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=callbacks
    )


def evaluate_scaled(model: keras.Model, test_data):
    X_test, y_test = test_data
    y_pred = model.predict(X_test, batch_size=BATCH_SIZE)

    print("\n=== Global Test Metrics (SCALED space) ===")
    for i, name in enumerate(TARGET_COLS):
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        rmse = float(np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i])))
        print(f"{name:9s} | MAE: {mae:.5f} | RMSE: {rmse:.5f}")


# ==============================
# 6) 27 plots: per center AND per meal (TEST only)
# ==============================
def plot_per_center_per_meal(df: pd.DataFrame, model: keras.Model):
    centers = sorted(df[GROUP_COL].unique())

    for cid in centers:
        g = df[df[GROUP_COL] == cid].sort_values(DATE_COL).reset_index(drop=True)
        if len(g) < (LOOKBACK + HORIZON + 10):
            print(f"[SKIP] Center {cid}: not enough rows")
            continue

        n = len(g)
        train_end, val_end = split_indices_per_group(n, TRAIN_RATIO, VAL_RATIO)

        # per-center scaler for targets (train only)
        y_raw = g[TARGET_COLS].to_numpy(dtype=np.float32)
        y_scaler = StandardScaler()
        y_scaler.fit(y_raw[:train_end])
        y_scaled = y_scaler.transform(y_raw)

        X_feat = g[FEATURE_COLS].to_numpy(dtype=np.float32)
        dates = g[DATE_COL].to_numpy(dtype="datetime64[ns]")

        X_all, y_all, y_dates = make_sequences(X_feat, y_scaled, dates, LOOKBACK, HORIZON)
        label_indices = np.arange(len(y_all)) + LOOKBACK + HORIZON - 1

        test_mask = label_indices >= val_end
        if not test_mask.any():
            print(f"[SKIP] Center {cid}: no test windows")
            continue

        X_test = X_all[test_mask]
        y_test_scaled = y_all[test_mask]
        test_dates = y_dates[test_mask]

        y_pred_scaled = model.predict(X_test, batch_size=BATCH_SIZE)

        # back to original units
        y_test = y_scaler.inverse_transform(y_test_scaled)  # (N,3)
        y_pred = y_scaler.inverse_transform(y_pred_scaled)  # (N,3)

        # --- one plot per meal (3 plots per center) ---
        for meal_i, meal_name in enumerate(TARGET_COLS):
            plt.figure(figsize=(14, 5))
            plt.plot(test_dates, y_test[:, meal_i], label="Actual")
            plt.plot(test_dates, y_pred[:, meal_i], label="Predicted", linestyle="--")
            plt.title(f"{meal_name} Forecast — RevenueCenterID {cid} (TEST)")
            plt.xlabel("Date")
            plt.ylabel("Revenue")
            plt.legend()
            plt.tight_layout()

            out_path = os.path.join(PLOTS_DIR, f"rc_{cid}_{meal_name}_test_pred_vs_actual.png")
            plt.savefig(out_path, dpi=150)
            plt.close()

            print(f"[OK] Saved -> {out_path}")


# ==============================
# 7) Main
# ==============================
def main():
    print("Loading dataset...")
    df = load_and_clean(CSV_PATH)
    print(f"Rows: {len(df)} | Centers: {df[GROUP_COL].nunique()}")
    print("Using features:", FEATURE_COLS)

    print("\nPreparing global train/val/test...")
    train_data, val_data, test_data = prepare_global_datasets(df)

    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    print("\nShapes:")
    print("  X_train:", X_train.shape, "y_train:", y_train.shape)  # y is (N,3)
    print("  X_val  :", X_val.shape, "y_val  :", y_val.shape)
    print("  X_test :", X_test.shape, "y_test :", y_test.shape)

    print("\nBuilding model...")
    model = build_model(input_dim=X_train.shape[-1])
    model.summary()

    print("\nTraining...")
    train_model(model, train_data, val_data)

    best_path = os.path.join(MODEL_DIR, "best_lstm.keras")
    if os.path.exists(best_path):
        model = keras.models.load_model(best_path)
        print(f"\nLoaded best model: {best_path}")

    print("\nEvaluating (scaled)...")
    evaluate_scaled(model, test_data)

    print("\nGenerating 27 plots (per center × per meal) ...")
    plot_per_center_per_meal(df, model)

    final_path = os.path.join(MODEL_DIR, "final_lstm.keras")
    model.save(final_path)
    print(f"\nSaved final model: {final_path}")
    print(f"Plots saved in: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
