import numpy as np
import pandas as pd

CATEGORIES = ["merch", "cat_id", "gender", "one_city", "us_state", "jobs"]


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def run_preproc(input_df):
    frame = input_df.copy()
    times = pd.to_datetime(frame["transaction_time"], errors="raise")
    frame["haversine"] = haversine(frame.lat, frame.lon, frame.merchant_lat, frame.merchant_lon)
    for part in ("year", "month", "day", "hour", "minute"):
        frame[part] = getattr(times.dt, part)
    frame["weekend"] = (times.dt.dayofweek >= 5).astype(int)
    hour = times.dt.hour + times.dt.minute / 60
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    frame["weekday_sin"] = np.sin(2 * np.pi * times.dt.dayofweek / 7)
    frame["weekday_cos"] = np.cos(2 * np.pi * times.dt.dayofweek / 7)
    frame["is_night"] = (times.dt.hour < 6).astype(int)
    frame["amount_log1p"] = np.log1p(frame.amount.clip(lower=0))
    frame["population_log1p"] = np.log1p(frame.population_city.clip(lower=0))
    frame["distance_log1p"] = np.log1p(frame.haversine)
    frame["amount_per_distance"] = frame.amount / (1 + frame.haversine)
    for column in CATEGORIES:
        frame[column] = frame[column].fillna("__missing__").astype(str)
    return frame.drop(columns=["transaction_time", "name_1", "name_2", "street"])
