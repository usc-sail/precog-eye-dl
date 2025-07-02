import pickle, json
from transformers import pipeline

sentiment_pipeline = pipeline(
    "text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

with open("metadata/all_sentences.pkl", "rb") as f:
    all_sentences = pickle.load(f)
    # get last word of each sentence
    all_last_words = [s.split()[-1][:-1] for s in all_sentences]

sentiment_s_scores = sentiment_pipeline(all_sentences)
sentiment_w_scores = sentiment_pipeline(all_last_words)

# organize them as a dict
sentiment_scores = {}
for i, s in enumerate(all_sentences):
    s_score = sentiment_s_scores[i]["label"]
    w_score = sentiment_w_scores[i]["label"]
    sentiment_scores[s] = [s_score, w_score]

with open("metadata/raw_sentiment.json", "w") as f:
    json.dump(sentiment_scores, f, indent=4)
