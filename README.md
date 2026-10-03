# Fake News Detector

A Flask web app that predicts whether a news article is **Real** or **Fake**, using TF-IDF and Logistic Regression (scikit-learn).

## Features
- Check any news text and get a Real/Fake result with a confidence percentage
- Shows the top words that pushed the prediction toward Real or Fake
- History page listing everything you checked
- Accuracy dashboard with test accuracy, precision, recall and F1, plus a chart of your own checks
- Input check that asks for at least 10 words

## Tech Stack
Python, Flask, scikit-learn, pandas, Chart.js

## Project Structure
```
app.py            Flask web app
train.py          Trains the model and saves metrics
utils.py          Shared text cleaning function
model/            Saved model, vectorizer and metrics
templates/        HTML pages
static/           CSS
data/             Dataset (not included, see below)
```

## Setup

1. Clone the repository and open the folder:
```
cd fake-news-detector
```

2. Create and activate a virtual environment, then install the libraries:
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
On Mac/Linux, use `source venv/bin/activate` instead of the second line.

3. Run the app:
```
python app.py
```
Open http://127.0.0.1:5000 in your browser.

## Dataset (needed only to retrain)

The dataset is not included because the files are large. To retrain the model:

1. Download the **Fake and Real News Dataset** from Kaggle (it contains `Fake.csv` and `True.csv`).
2. Create a folder named `data` in the project and put both files inside it.
3. Run:
```
python train.py
```
Training takes a few minutes and creates `model/model.pkl`, `model/vectorizer.pkl` and `model/metrics.json`.

## Results
Test accuracy of about 97.6% on held-out data (80/20 split).



