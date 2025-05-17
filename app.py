from flask import Flask, request, jsonify
import pandas as pd
import requests

app = Flask(__name__)

HF_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
HF_TOKEN = "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" # Replace with your real Hugging Face token
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


def get_sentiment(text):
    try:
        response = requests.post(HF_URL, headers=HEADERS, json={"inputs": text})
        response.raise_for_status()
        result = response.json()[0]
        label = result['label']
        score = result['score']
        score = score if label == 'POSITIVE' else -score
        return label, round(score, 3)
    except Exception as e:
        return "ERROR", 0


def process_reviews(df):
    results = []
    total_score = 0

    for review in df['Review']:
        label, score = get_sentiment(str(review))
        results.append({"review": review, "sentiment": label, "score": score})
        total_score += score

    avg_score = round(total_score / len(results), 3)

    return {
        "average_sentiment_score": avg_score,
        "summary": f"Overall sentiment: {'Positive' if avg_score > 0 else 'Negative' if avg_score < 0 else 'Neutral'}",
        "top_3_reviews": results[:3]
    }


@app.route("/analyze", methods=["POST"])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "CSV file is required."}), 400

    csv_file = request.files['file']
    df = pd.read_csv(csv_file)

    if 'review' not in df.columns:
        return jsonify({"error": "CSV must have a 'review' column."}), 400

    return jsonify(process_reviews(df))


@app.route("/demo", methods=["GET"])
def demo():
    df = pd.read_csv("Customer_Reviews_for_Headsets.csv")
    return jsonify(process_reviews(df))


@app.route("/", methods=["GET"])
def root():
    return "Sentiment Analysis API is running! Use /analyze or /demo"
