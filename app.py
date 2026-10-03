from flask import Flask, render_template, request, redirect, Response
import os
import io
import csv
import json
import pickle
import sqlite3
from datetime import datetime
from utils import clean_text

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "history.db")

model = pickle.load(open(os.path.join(BASE, "model", "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE, "model", "vectorizer.pkl"), "rb"))

with open(os.path.join(BASE, "model", "metrics.json")) as f:
    metrics = json.load(f)

feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_[0]


# ---------------- DATABASE (SQLite) ----------------

def db_execute(query, params=()):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(query, params)
    conn.commit()
    conn.close()


def db_query(query, params=()):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return rows


db_execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        news TEXT,
        result TEXT,
        confidence REAL,
        created_at TEXT
    )
""")


# ---------------- EXPLAIN PREDICTION ----------------

def explain(vector, n=5):
    row = vector.tocoo()
    contrib = [(feature_names[j], v * coefs[j]) for j, v in zip(row.col, row.data)]
    real = sorted([c for c in contrib if c[1] > 0], key=lambda x: -x[1])[:n]
    fake = sorted([c for c in contrib if c[1] < 0], key=lambda x: x[1])[:n]
    return [w for w, _ in real], [w for w, _ in fake]


# ---------------- HOME PAGE ----------------

@app.route("/")
def home():
    return render_template("home.html")


# ---------------- CHECK NEWS PAGE ----------------

@app.route("/check", methods=["GET", "POST"])
def check():

    result = ""
    confidence = 0
    error = ""
    real_words, fake_words = [], []

    if request.method == "POST":

        news = request.form["news"].strip()

        if len(news.split()) < 10:
            error = "Please enter at least 10 words for a reliable result."
        else:
            cleaned = clean_text(news)

            vector = vectorizer.transform([cleaned])

            prediction = model.predict(vector)

            prob = model.predict_proba(vector)

            confidence = float(round(max(prob[0]) * 100, 2))

            if prediction[0] == 1:
                result = "Real News"
            else:
                result = "Fake News"

            real_words, fake_words = explain(vector)

            db_execute(
                "INSERT INTO history (news, result, confidence, created_at) VALUES (?, ?, ?, ?)",
                (news, result, confidence, datetime.now().strftime("%d-%m-%Y %H:%M"))
            )

    return render_template(
        "check_news.html",
        result=result,
        confidence=confidence,
        error=error,
        real_words=real_words,
        fake_words=fake_words
    )


# ---------------- HISTORY PAGE ----------------

@app.route("/history")
def history_page():

    rows = db_query(
        "SELECT id, news, result, confidence, created_at FROM history ORDER BY id DESC"
    )

    return render_template("history.html", history=rows)


# ---------------- DOWNLOAD HISTORY AS CSV ----------------

@app.route("/download")
def download():

    rows = db_query(
        "SELECT news, result, confidence, created_at FROM history ORDER BY id DESC"
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["News", "Result", "Confidence (%)", "Time"])
    writer.writerows(rows)

    # "\ufeff" helps Excel display special characters correctly
    return Response(
        "\ufeff" + output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=history.csv"}
    )


# ---------------- DELETE ONE HISTORY ROW ----------------

@app.route("/delete/<int:item_id>", methods=["POST"])
def delete(item_id):

    db_execute("DELETE FROM history WHERE id = ?", (item_id,))

    return redirect("/history")


# ---------------- ACCURACY DASHBOARD ----------------

@app.route("/accuracy")
def accuracy():

    counts = dict(db_query("SELECT result, COUNT(*) FROM history GROUP BY result"))

    return render_template(
        "accuracy.html",
        real_count=counts.get("Real News", 0),
        fake_count=counts.get("Fake News", 0),
        metrics=metrics
    )


# ---------------- CLEAR HISTORY ----------------

@app.route("/clear", methods=["POST"])
def clear():

    db_execute("DELETE FROM history")

    return redirect("/history")


# ---------------- RUN APP ----------------

if __name__ == "__main__":
    app.run(debug=True)