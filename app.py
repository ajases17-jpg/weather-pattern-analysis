raise ValueError("Rainfall must be between 0 and 500 mm.")
        if not 0 <= humidity <= 100:
            raise ValueError("Humidity must be between 0 and 100%.")
        if not 0 <= wind_speed <= 300:
            raise ValueError("Wind speed must be between 0 and 300 km/h.")
        if not 1 <= month <= 12:
            raise ValueError("Month must be from 1 to 12.")
        if not 0 <= hour <= 23:
            raise ValueError("Hour must be from 0 to 23.")

        input_data = pd.DataFrame([{
            "humidity": humidity,
            "rainfall": rainfall,
            "wind_speed": wind_speed,
            "month": month,
            "hour": hour
        }])

        predicted_temperature = float(model.predict(input_data)[0])
        difference = abs(temperature - predicted_temperature)

        if predicted_temperature >= 35:
            category = "Hot"
        elif predicted_temperature >= 25:
            category = "Warm"
        else:
            category = "Cool"

        return render_template(
            "index.html",
            graph_data=graph_data,
            mae=mae,
            prediction=round(predicted_temperature, 2),
            difference=round(difference, 2),
            category=category,
            error=None
        )

    except (ValueError, KeyError):
        return render_template(
            "index.html",
            graph_data=graph_data,
            mae=mae,
            prediction=None,
            difference=None,
            category=None,
            error="Please enter valid weather values."
        ), 400


if name == "main":
    app.run(debug=True)
