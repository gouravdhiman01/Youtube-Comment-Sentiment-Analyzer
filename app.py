from flask import Flask, render_template, request
from fetch_analyze import analyze_youtube_video
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        video_url = request.form.get("video_url")

        try:
            result = analyze_youtube_video(video_url)
        except Exception as e:
            error = str(e)

    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    app.run(debug=True)
