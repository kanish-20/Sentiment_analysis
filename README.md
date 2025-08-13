# Sentiment Analysis Web App (Naive Bayes)

This project is a web-based sentiment analysis application built with Python, Flask, and scikit-learn. It predicts whether a given text review is positive, negative, or neutral using a **Naive Bayes** classifier trained on a custom dataset.

## Features

- Classifies text reviews into positive, negative, or neutral sentiments.

- Simple and user-friendly web interface built with Flask.

- Preprocessing: Converts text to lowercase and removes special characters.

- Uses CountVectorizer for text feature extraction.

- Supports custom datasets for training the model.

## Project Structure

```
sentiment_analysis/
│
├─ app.py                     # Flask web application
├─ train_model.py             # Script to train Naive Bayes model
├─ sentiment_data.csv         # Dataset of reviews
├─ sentiment_nb_model.pkl     # Saved trained model (generated after training)
├─ sentiment_vectorizer.pkl   # Saved vectorizer (generated after training)
├─ requirements.txt           # Python dependencies
├─ templates/
│   └─ index.html             # HTML page for user input
└─ static/
    └─ style.css              # Styling for the web app
```

## Installation

### 1.Clone the repository:
```
git clone https://github.com/yourusername/sentiment_analysis.git
cd sentiment_analysis
```

### 3.Install dependencies:
```
pip install -r requirements.txt
```

## Usage

### 1. Train the Model


