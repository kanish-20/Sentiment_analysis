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
```
python train_model.py
```

### 2.Run the Flask App
```
python app.py
```
## Open your browser and go to:
```
http://127.0.0.1:5000/
```

## Future Improvements

- Use TF-IDF Vectorizer for better feature representation.

- Expand dataset to include more diverse reviews for higher accuracy.

- Add color-coded output for sentiments (green=positive, red=negative, yellow=neutral).

- Replace Naive Bayes with more advanced models like Logistic Regression, Random Forest, or BERT for better predictions.

## Dependencies

- Python 3.11+

- Flask

- scikit-learn

- pandas

- joblib

## Screenshots

![WhatsApp Image 2025-08-13 at 12 22 05 (2)](https://github.com/user-attachments/assets/374ba8e2-7378-42fa-bdf5-bf2b7625506e)
![WhatsApp Image 2025-08-13 at 19 26 22](https://github.com/user-attachments/assets/9eb5f849-031c-43a1-b172-9d0269c99206)
![WhatsApp Image 2025-08-13 at 19 26 22 (1)](https://github.com/user-attachments/assets/d907f909-c5e5-4ea9-a1e7-ce5a6d9a1664)


