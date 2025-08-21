# type:ignore

# importing libraries
import re
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
import joblib

#  Text Cleaning 
sws = list(ENGLISH_STOP_WORDS)
for stop in ["not", "no"]:
    if stop in sws:
        sws.remove(stop)
sws = [w for w in sws if not w.endswith("n't")]

def text_cleaning(doc):
    newdoc = re.sub(r'[^a-z0-9\s]', '', doc.lower())
    words = newdoc.split()
    newdoc = [word for word in words if word not in sws]
    return ' '.join(newdoc)

# Load Dataset 
df = pd.read_csv("reviews_data_2000.csv")
corpus = df["Review"].apply(text_cleaning)
target = df["Liked"]

#  Vectorization 
cv = CountVectorizer(binary=True)
X = cv.fit_transform(corpus).toarray()

#  Model ( we can use more model but we get best score in bernoulli)
model = BernoulliNB()

pipeline = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("model", BernoulliNB())
])

#  MLflow Experiment Name
mlflow.set_experiment("Sentiment_Analysis")

with mlflow.start_run(run_name="BernoulliNB_CountVectorizer"):
    scores = cross_validate(pipeline, corpus, target, cv=5, return_train_score=True)
    train_score = scores["train_score"].mean()
    test_score = scores["test_score"].mean()

    # Log params & metrics
    mlflow.log_param("model", "BernoulliNB")
    mlflow.log_metric("train_score", train_score)
    mlflow.log_metric("test_score", test_score)

    # Train final pipeline
    pipeline.fit(corpus, target)

    # Save with joblib
    joblib.dump(pipeline, "sentiment_model.pkl")

    # Log to MLflow
    mlflow.sklearn.log_model(pipeline, "sentiment_pipeline")

    print("✅ Model trained and logged")
    print(f"Train: {train_score:.4f}, Test: {test_score:.4f}")

