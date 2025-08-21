import joblib
model = joblib.load("sentiment_model.pkl")

sample = ["I did not like the food", "Wow, absolutely loved it!"]
preds = model.predict(sample)

print(preds)  # [0, 1]
