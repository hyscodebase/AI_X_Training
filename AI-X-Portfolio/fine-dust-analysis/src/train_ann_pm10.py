#!/usr/bin/env python3
"""Train an ANN model to predict next-hour PM10 using four fine-dust datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
RANDOM_STATE = 42


def parse_air_time(series: pd.Series) -> pd.Series:
    """Convert air timestamps from 1-24 hour format to hourly datetimes."""
    ts = series.astype(str).str.zfill(10)
    dates = pd.to_datetime(ts.str[:8], format="%Y%m%d")
    hours = ts.str[8:].astype(int)
    return dates + pd.to_timedelta(hours, unit="h") - pd.Timedelta(hours=1)


def load_air(year: int) -> pd.DataFrame:
    air = pd.read_csv(DATA_DIR / f"air_{year}.csv")
    air["time"] = parse_air_time(air["측정일시"])
    return air.sort_values("time").reset_index(drop=True)


def load_weather(year: int) -> pd.DataFrame:
    weather = pd.read_csv(DATA_DIR / f"weather_{year}.csv", encoding="cp949")
    weather["time"] = pd.to_datetime(weather["일시"])
    return weather.sort_values("time").reset_index(drop=True)


def merge_year(year: int) -> pd.DataFrame:
    air = load_air(year)
    weather = load_weather(year)
    merged = pd.merge(air, weather, on="time", how="inner")
    return merged.sort_values("time").reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    hour = out["time"].dt.hour
    day_of_week = out["time"].dt.dayofweek
    day_of_year = out["time"].dt.dayofyear
    year_length = np.where(out["time"].dt.is_leap_year, 366, 365)

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    out["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    out["doy_sin"] = np.sin(2 * np.pi * day_of_year / year_length)
    out["doy_cos"] = np.cos(2 * np.pi * day_of_year / year_length)
    out["is_weekend"] = (day_of_week >= 5).astype(int)

    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    lag_map = {
        "PM10": [1, 3, 6, 24],
        "PM25": [1, 3, 24],
        "기온(°C)": [1, 24],
        "풍속(m/s)": [1, 24],
        "습도(%)": [1, 24],
    }
    for column, lags in lag_map.items():
        for lag in lags:
            out[f"{column}_lag_{lag}h"] = out[column].shift(lag)

    out["PM10_roll_mean_6h"] = out["PM10"].rolling(window=6, min_periods=1).mean()
    out["PM10_roll_mean_24h"] = out["PM10"].rolling(window=24, min_periods=1).mean()
    out["PM10_roll_std_24h"] = out["PM10"].rolling(window=24, min_periods=2).std()
    out["PM25_roll_mean_24h"] = out["PM25"].rolling(window=24, min_periods=1).mean()

    if {"풍향(16방위)", "풍속(m/s)"}.issubset(out.columns):
        radians = np.deg2rad(out["풍향(16방위)"])
        out["wind_u"] = out["풍속(m/s)"] * np.cos(radians)
        out["wind_v"] = out["풍속(m/s)"] * np.sin(radians)

    return out

def fill_domain_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    zero_fill_columns = [
        "강수량(mm)",
        "적설(cm)",
        "3시간신적설(cm)",
        "일조(hr)",
        "일사(MJ/m2)",
        "전운량(10분위)",
        "중하층운량(10분위)",
    ]
    for column in zero_fill_columns:
        if column in out.columns:
            out[column] = out[column].fillna(0)
    return out


def prepare_dataset(year: int) -> pd.DataFrame:
    df = merge_year(year)
    df = fill_domain_missing_values(df)
    df = add_time_features(df)
    df = add_lag_features(df)
    df["PM10_target_t_plus_1"] = df["PM10"].shift(-1)
    return df.sort_values("time").reset_index(drop=True)


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_drop_columns = {
        "time",
        "PM10_target_t_plus_1",
        "Unnamed: 0",
        "지역",
        "망",
        "측정소명",
        "주소",
        "측정일시",
        "지점",
        "지점명",
        "일시",
        "운형(운형약어)",
        "지면상태(지면상태코드)",
        "현상번호(국내식)",
        "측정소코드",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in feature_drop_columns
        and "QC플래그" not in column
        and pd.api.types.is_numeric_dtype(df[column])
    ]
    x = df[feature_columns].copy()
    y = df["PM10_target_t_plus_1"].copy()
    valid_rows = y.notna()
    return x.loc[valid_rows].reset_index(drop=True), y.loc[valid_rows].reset_index(drop=True)


def align_feature_frames(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    threshold: float = 0.98,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sparse_columns = train_x.columns[train_x.isna().mean() >= threshold].tolist()
    train_aligned = train_x.drop(columns=sparse_columns)
    test_aligned = test_x.drop(columns=sparse_columns, errors="ignore")
    return train_aligned, test_aligned.reindex(columns=train_aligned.columns)


def make_pipeline(hidden_layer_sizes: tuple[int, ...], alpha: float, learning_rate_init: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation="relu",
                    solver="adam",
                    alpha=alpha,
                    batch_size=256,
                    learning_rate_init=learning_rate_init,
                    max_iter=600,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=30,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def select_best_model(x_train: pd.DataFrame, y_train: pd.Series) -> tuple[Pipeline, dict[str, object]]:
    split_index = int(len(x_train) * 0.8)
    x_fit, x_val = x_train.iloc[:split_index], x_train.iloc[split_index:]
    y_fit, y_val = y_train.iloc[:split_index], y_train.iloc[split_index:]

    candidates = [
        {"hidden_layer_sizes": (64, 32), "alpha": 1e-4, "learning_rate_init": 1e-3},
        {"hidden_layer_sizes": (128, 64, 32), "alpha": 5e-4, "learning_rate_init": 7e-4},
        {"hidden_layer_sizes": (256, 128, 64), "alpha": 1e-3, "learning_rate_init": 5e-4},
    ]

    best_pipeline: Optional[Pipeline] = None
    best_config: Optional[dict[str, object]] = None
    best_rmse = np.inf

    for config in candidates:
        pipeline = make_pipeline(**config)
        pipeline.fit(x_fit, y_fit)
        val_predictions = pipeline.predict(x_val)
        candidate_rmse = rmse(y_val, val_predictions)
        if candidate_rmse < best_rmse:
            best_rmse = candidate_rmse
            best_pipeline = pipeline
            best_config = {**config, "validation_rmse": candidate_rmse}

    assert best_pipeline is not None and best_config is not None
    final_pipeline = clone(best_pipeline)
    final_pipeline.fit(x_train, y_train)
    return final_pipeline, best_config


def save_dataframe(df: Union[pd.DataFrame, pd.Series], path: Path) -> None:
    if isinstance(df, pd.Series):
        df.to_frame().to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = prepare_dataset(2021)
    test_df = prepare_dataset(2022)

    train_x, train_y = build_feature_frame(train_df)
    test_x, test_y = build_feature_frame(test_df)
    train_x, test_x = align_feature_frames(train_x, test_x)

    model, best_config = select_best_model(train_x, train_y)
    train_predictions = model.predict(train_x)
    test_predictions = model.predict(test_x)

    metrics = {
        "train": evaluate(train_y, train_predictions),
        "test": evaluate(test_y, test_predictions),
        "selected_model": best_config,
        "train_rows": int(len(train_x)),
        "test_rows": int(len(test_x)),
        "feature_count": int(train_x.shape[1]),
    }

    feature_info = {
        "feature_columns": train_x.columns.tolist(),
    }

    prediction_frame = pd.DataFrame(
        {
            "actual_pm10_t_plus_1": test_y,
            "predicted_pm10_t_plus_1": test_predictions,
        }
    )

    save_dataframe(train_x, ARTIFACTS_DIR / "train_x.csv")
    save_dataframe(train_y.rename("PM10_target_t_plus_1"), ARTIFACTS_DIR / "train_y.csv")
    save_dataframe(test_x, ARTIFACTS_DIR / "test_x.csv")
    save_dataframe(test_y.rename("PM10_target_t_plus_1"), ARTIFACTS_DIR / "test_y.csv")
    save_dataframe(prediction_frame, ARTIFACTS_DIR / "test_predictions.csv")

    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    with open(ARTIFACTS_DIR / "feature_columns.json", "w", encoding="utf-8") as file:
        json.dump(feature_info, file, ensure_ascii=False, indent=2)

    joblib.dump(model, ARTIFACTS_DIR / "pm10_ann_pipeline.joblib")

    print("Saved artifacts to:", ARTIFACTS_DIR)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
