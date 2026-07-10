import subprocess
subprocess.run(["pip","install","joblib"])
import streamlit as st
import requests
import pandas as pd
import joblib
from datetime import datetime


# 모델 불러오기
@st.cache_resource
def load_model():
    return joblib.load("solar_model(6).pkl")


model = load_model()


# 인천 좌표
LAT = 37.4563
LON = 126.7052


# Open-Meteo 실시간 날씨 가져오기
def get_weather():

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": LAT,
        "longitude": LON,
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "cloud_cover"
        ],
        "timezone": "Asia/Seoul"
    }


    response = requests.get(
        url,
        params=params
    )

    data = response.json()

    return data["current"]



# 화면

st.title("☀ 인천 태양광 일사량 예측 시스템")

st.write(
    "실시간 기상 데이터를 이용하여 예상 일사량을 계산합니다."
)


# 날씨 받아오기

weather = get_weather()


temperature = weather["temperature_2m"]
humidity = weather["relative_humidity_2m"]
wind_speed = weather["wind_speed_10m"]
cloud = weather["cloud_cover"]



st.subheader("현재 인천 기상 정보")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "기온",
        f"{temperature} ℃"
    )

    st.metric(
        "습도",
        f"{humidity} %"
    )


with col2:
    st.metric(
        "풍속",
        f"{wind_speed} km/h"
    )

    st.metric(
        "운량",
        f"{cloud} %"
    )



# 모델 입력

input_data = pd.DataFrame(
    [[
        temperature,
        humidity,
        wind_speed,
        cloud
    ]],
    columns=[
        "temperature",
        "humidity",
        "wind_speed",
        "cloud"
    ]
)



# 예측

prediction = model.predict(input_data)[0]


# 음수 방지

prediction = max(0, prediction)



st.divider()

st.subheader("☀ 예상 일사량")

st.metric(
    "Solar Radiation",
    f"{prediction:.2f} MJ/m²"
)



now = datetime.now()

st.caption(
    f"업데이트 시간 : {now.strftime('%Y-%m-%d %H:%M')}"
)
