from flask import Flask, render_template, request
import pandas as pd
import requests

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)

model = None
mae = None

graph_data = {
    "labels": [],
    "temperature": [],
    "rainfall": [],
    "humidity": []
}


def train_model():
    global model, mae, graph_data

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": 9.9312,
        "longitude": 76.2673,
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m"
        ],
        "timezone": "Asia/Kolkata"
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    weather = response.json()["hourly"]

    df = pd.DataFrame({
        "time": pd.to_datetime(weather["time"]),
        "temperature": weather["temperature_2m"],
        "humidity": weather["relative_humidity_2m"],
        "rainfall": weather["precipitation"],
        "wind_speed": weather["wind_speed_10m"]
    })

    df = df.dropna().copy()
    df["month"] = df["time"].dt.month
    df["hour"] = df["time"].dt.hour

    features = [
        "humidity",
        "rainfall",
        "wind_speed",
        "month",
        "hour"
    ]

    X = df[features]
    y = df["temperature"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)
    mae = round(
        mean_absolute_error(y_test, test_predictions), 2
    )

    chart = df.head(168)

    graph_data = {
        "labels": chart["time"].dt.strftime("%d %H:%M").tolist(),
        "temperature": chart["temperature"].round(2).tolist(),
        "rainfall": chart["rainfall"].round(2).tolist(),
        "humidity": chart["humidity"].tolist()
    }


def show_page(prediction=None, difference=None,
              category=None, error=None, status=200):
    return render_template(
        "index.html",
        graph_data=graph_data,
        mae=mae,
        prediction=prediction,
        difference=difference,
        category=category,
        error=error
    ), status


@app.route("/")
def home():
    try:
        if model is None:
            train_model()
        return show_page()
    except Exception:
        app.logger.exception("Model setup failed")
        return (
            "Weather model setup failed. Please try again later.",
            503
        )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            train_model()

        temperature = float(request.form["temperature"])
        rainfall = float(request.form["rainfall"])
        humidity = float(request.form["humidity"])
        wind_speed = float(request.form["wind_speed"])
        month = int(request.form["month"])
        hour = int(request.form["hour"])

        if not -50 <= temperature <= 60:
            raise ValueError("Invalid temperature.")
        if not 0 <= rainfall <= 500:
            raise ValueError("Rainfall must be between 0 and 500 mm.")
        if not 0 <= humidity <= 100:
            raise ValueError("Humidity must be between 0 and 100.")
        if not 0 <= wind_speed <= 300:
            raise ValueError("Invalid wind speed.")
        if not 1 <= month <= 12:
            raise ValueError("Month must be from 1 to 12.")
        if not 0 <= hour <= 23:
            raise ValueError("Hour must be from 0 to 23.")

        inputs = pd.DataFrame([{
            "humidity": humidity,
            "rainfall": rainfall,
            "wind_speed": wind_speed,
            "month": month,
            "hour": hour
        }])
        
        predicted = round(float(model.predict(inputs)[0]), 2)
        difference = round(abs(temperature - predicted), 2)

        if predicted >= 35:
            category = "Hot"
        elif predicted >= 25:
            category = "Warm"
        else:
            category = "Cool"

        return show_page(
            prediction=predicted,
            difference=difference,
            category=category
        )

    except ValueError as error:
        return show_page(error=str(error), status=400)
    except Exception:
        app.logger.exception("Prediction failed")
        return show_page(
            error="Prediction failed. Please try again.",
            status=500
        )


if __name__ == "__main__":
    app.run()
