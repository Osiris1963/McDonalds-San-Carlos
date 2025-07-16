# run_forecast.py

import os
from firebase_admin import credentials, initialize_app, firestore
from datetime import datetime

# 1) Init Firestore
if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    initialize_app(credentials.Certificate(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")))
else:
    initialize_app()
db = firestore.client()

def main():
    # 2) Load historical data
    docs = db.collection("sample_historical_data").stream()
    data = [d.to_dict() for d in docs]

    # 3) TODO: ibutang dinhi imong forecasting logic
    #    e.g. forecast = your_forecast_function(data)
    forecast = {"example": 123}

    # 4) Save result
    db.collection("forecast_log").add({
        "run_at": firestore.SERVER_TIMESTAMP,
        "result": forecast
    })
    print(f"[{datetime.now()}] Forecast run complete.")

if __name__ == "__main__":
    main()
