from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

# -------- Config --------
MODEL_PATH = os.getenv("MODEL_PATH", "podcast_model.pkl")

# Load model once at startup
model = joblib.load(MODEL_PATH)

app = Flask(__name__)

# Home page with form
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Form POST -> prediction
@app.route("/predict", methods=["POST"])
def predict_form():
    try:
        payload = {
            "Episode_Length_minutes": float(request.form.get("Episode_Length_minutes", 0)),
            "Host_Popularity_percentage": float(request.form.get("Host_Popularity_percentage", 0)),
            "Number_of_Ads": int(request.form.get("Number_of_Ads", 0)),
            "Genre": request.form.get("Genre", "").strip(),
            "Publication_Day": request.form.get("Publication_Day", "").strip(),
            "Publication_Time": request.form.get("Publication_Time", "").strip(),
            "Episode_Sentiment": request.form.get("Episode_Sentiment", "").strip(),
        }
        X = pd.DataFrame([payload])
        pred = float(model.predict(X)[0])
        return render_template("index.html", prediction=f"{pred:.2f} minutes", last_input=payload)
    except Exception as e:
        return render_template("index.html", error=str(e))

# JSON API
@app.route("/api/predict", methods=["POST"])
def predict_api():
    """
    Accepts JSON like:
    {
      "Episode_Length_minutes": 60,
      "Host_Popularity_percentage": 75.0,
      "Number_of_Ads": 2,
      "Genre": "Health",
      "Publication_Day": "Friday",
      "Publication_Time": "Evening",
      "Episode_Sentiment": "Positive"
    }
    """
    try:
        data = request.get_json(force=True)

        required = [
            "Episode_Length_minutes",
            "Host_Popularity_percentage",
            "Number_of_Ads",
            "Genre",
            "Publication_Day",
            "Publication_Time",
            "Episode_Sentiment",
        ]
        missing = [k for k in required if k not in data]
        if missing:
            return jsonify({"error": f"Missing field(s): {', '.join(missing)}"}), 400

        # Coerce basic types
        clean = {
            "Episode_Length_minutes": float(data["Episode_Length_minutes"]),
            "Host_Popularity_percentage": float(data["Host_Popularity_percentage"]),
            "Number_of_Ads": int(data["Number_of_Ads"]),
            "Genre": str(data["Genre"]).strip(),
            "Publication_Day": str(data["Publication_Day"]).strip(),
            "Publication_Time": str(data["Publication_Time"]).strip(),
            "Episode_Sentiment": str(data["Episode_Sentiment"]).strip(),
        }

        X = pd.DataFrame([clean])
        pred = float(model.predict(X)[0])
        return jsonify({"prediction_minutes": round(pred, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # For local dev only; use gunicorn/uvicorn in production
    app.run(host="0.0.0.0", port=5000, debug=True)
