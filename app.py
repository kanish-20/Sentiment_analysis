from flask import Flask, request, render_template
import joblib
import re

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("sentiment_nb_model.pkl")
vectorizer = joblib.load("sentiment_vectorizer.pkl")

# Text cleaning function (must match training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    review_text = ""
    if request.method == 'POST':
        review_text = request.form['review']
        review_cleaned = clean_text(review_text)
        review_vectorized = vectorizer.transform([review_cleaned])
        prediction = model.predict(review_vectorized)[0]
        prediction_text = f'Sentiment: {prediction}'
    return render_template('index.html', prediction_text=prediction_text, review=review_text)

if __name__ == "__main__":
    app.run(debug=True)
