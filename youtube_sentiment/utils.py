import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words("english"))
vader = SentimentIntensityAnalyzer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)
    text = text.strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

def sentiment_label(text):
    v_score = vader.polarity_scores(text)["compound"]
    tb_score = TextBlob(text).sentiment.polarity
    score = (0.6 * v_score) + (0.4 * tb_score)

    if score >= 0.05:
        return "positive", score
    elif score <= -0.05:
        return "negative", score
    else:
        return "neutral", score
