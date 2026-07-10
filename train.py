import requests
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# 인천 좌표
LAT = 37.4563
LON = 126.7052


# 과거 데이터 다운로드
url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": "2020-01-01",
    "end_date": "2025-12-31",
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "cloud_cover",
        "shortwave_radiation"
    ],
    "timezone": "Asia/Seoul"
}


response = requests.get(
    url,
    params=params
)


data = response.json()


# dataframe 변환

df = pd.DataFrame(data["hourly"])


df = df.rename(
    columns={
        "temperature_2m":"temperature",
        "relative_humidity_2m":"humidity",
        "wind_speed_10m":"wind_speed",
        "cloud_cover":"cloud",
        "shortwave_radiation":"solar"
    }
)


# 결측 제거

df = df.dropna()


print(df.head())


# 입력 변수

X = df[
[
"temperature",
"humidity",
"wind_speed",
"cloud"
]
]


# 목표값

y = df["solar"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


model.fit(
    X_train,
    y_train
)


print(
    "학습 완료"
)


print(
    "정확도:",
    model.score(X_test,y_test)
)


joblib.dump(
    model,
    "solar_model.pkl"
)

print(
    "모델 저장 완료"
)
