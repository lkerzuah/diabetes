import joblib

def predict_diabetes(input_df):
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('model.pkl')
    feature_list = joblib.load('features.pkl')

    try:
        input_df = input_df[feature_list]
    except KeyError:
        missing = set(feature_list) - set(input_df.columns)
        return {"error": f"Missing required features: {missing}"}

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled)[0][prediction]

    label = "Diabetic" if prediction == 1 else "Non-Diabetic"

    return {
        "label": label,
        "confidence": round(confidence * 99, 1),
        "input": input_df.to_dict(orient="records")[0]
    }
