import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load expanded dataset
df = pd.read_csv("sentiment_data.csv",on_bad_lines='skip')

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['review'] = df['review'].apply(clean_text)

# Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "sentiment_nb_model.pkl")
joblib.dump(vectorizer, "sentiment_vectorizer.pkl")
print("Model and vectorizer saved successfully!")
