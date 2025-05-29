from flask import Flask, request, jsonify
import pandas as pd
import requests
import os
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)  # Logs all levels DEBUG and above

HF_URL = "https://api-inference.huggingface.co/models/distilbert/distilbert-base-uncased-finetuned-sst-2-english"
HF_TOKEN = "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" # Replace with your real Hugging Face token
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}


def get_sentiment(text):
    app.logger.debug(f"Calling get_sentiment for text divya: {text}")
    try:
        response = requests.post(HF_URL, headers=HEADERS, json={"inputs": text})
         
        app.logger.debug(f"Raw response status: {response.status_code}")
        app.logger.debug(f"Raw response text: {response.text}")
        response.raise_for_status()
        result = response.json()[0][0]
        label = result['label']
        score = result['score']
        score = score if label == 'POSITIVE' else -score
        app.logger.debug(f"Sentiment: {label}, Score: {score}")
        return label, round(score, 3)
    except Exception as e:
        app.logger.error(f"Error in get_sentiment: {e}")
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


@app.route("/demo/<model>", methods=["GET"])
def demo(model):
    app.logger.debug(f"Requested headset model: {model}")

    df = pd.read_csv("Consolidated_MacBook_Pro_Reviews.csv")

    if 'Headset Model' not in df.columns or 'Review' not in df.columns:
        return jsonify({"error": "CSV must have 'Model' and 'Review' columns."}), 400

    filtered_df = df[df['Model'].str.lower() == model.lower()]

    if filtered_df.empty:
        return jsonify({"error": f"No reviews found for model: {model}"}), 404

    return jsonify(process_reviews(filtered_df))


@app.route("/", methods=["GET"])
def root():
    return "Sentiment Analysis API is running! Use /analyze or /demo"

@app.route("/health")
def health():
    return "OK", 200


# 🔧 Render requires you to bind to PORT from env variable
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # fallback to 5000 for local testing
    app.run(host="0.0.0.0", port=port)
