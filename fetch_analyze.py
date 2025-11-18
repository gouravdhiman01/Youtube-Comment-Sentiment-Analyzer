import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
from textblob import TextBlob

# Load API Key from environment variable
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


# -------------------------------
# EXTRACT VIDEO ID FROM URL
# -------------------------------
def get_video_id(url):
    """
    Extracts the video ID from multiple YouTube URL formats.
    """
    pattern = r"(?:v=|\.be/|embed/)([A-Za-z0-9_-]{11})"
    match = re.search(pattern, url)
    if not match:
        raise ValueError("Invalid YouTube URL. Please enter a correct link.")
    return match.group(1)


# -------------------------------
# FETCH COMMENTS USING YOUTUBE API
# -------------------------------
def fetch_comments(video_id):
    """
    Fetches top-level comments from YouTube using the official API.
    """

    if not YOUTUBE_API_KEY:
        raise Exception("YouTube API key missing! Set YOUTUBE_API_KEY in Render Environment Variables.")

    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    comments = []
    next_page_token = None

    while True:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token,
            order="relevance"
        ).execute()

        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        next_page_token = response.get("nextPageToken")

        if not next_page_token or len(comments) >= 300:
            break

    return comments


# -------------------------------
# PERFORM SENTIMENT ANALYSIS
# -------------------------------
def analyze_sentiments(comments):
    """
    Uses TextBlob polarity score to classify comment sentiment.
    """

    sentiment_labels = []
    for text in comments:
        score = TextBlob(text).sentiment.polarity

        if score > 0.1:
            label = "Positive"
        elif score < -0.1:
            label = "Negative"
        else:
            label = "Neutral"

        sentiment_labels.append(label)

    df = pd.DataFrame({
        "Comment": comments,
        "Sentiment": sentiment_labels
    })

    return df


# -------------------------------
# GENERATE PIE & BAR CHARTS
# -------------------------------
def generate_charts(df):
    """
    Creates pie chart and bar chart, saves in /static folder.
    """
    sentiment_counts = df["Sentiment"].value_counts()

    # PIE CHART
    plt.figure(figsize=(5, 4))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%")
    pie_path = "static/pie_chart.png"
    plt.savefig(pie_path)
    plt.close()

    # BAR CHART
    plt.figure(figsize=(5, 4))
    sentiment_counts.plot(kind="bar")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    bar_path = "static/bar_chart.png"
    plt.savefig(bar_path)
    plt.close()

    return pie_path, bar_path


# -------------------------------
# MAIN FUNCTION CALLED BY app.py
# -------------------------------
def analyze_youtube_video(video_url):
    """
    Complete pipeline:
    1. Extract video ID
    2. Fetch comments
    3. Analyze sentiments
    4. Save CSV
    5. Generate charts
    6. Return paths
    """

    video_id = get_video_id(video_url)
    comments = fetch_comments(video_id)
    df = analyze_sentiments(comments)

    # Save CSV
    csv_path = "static/comments_sentiment.csv"
    df.to_csv(csv_path, index=False)

    # Generate Charts
    pie, bar = generate_charts(df)

    return {
        "csv": csv_path,
        "pie": pie,
        "bar": bar
    }
