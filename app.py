import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Initialize Flask app
app = Flask(__name__)

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Sample data
data = {
    'Movie': ['Movie 1', 'Movie 2', 'Movie 3', 'Movie 4', 'Movie 5',
              'Movie 6', 'Movie 7', 'Movie 8', 'Movie 9', 'Movie 10',
              'Movie 11', 'Movie 12', 'Movie 13', 'Movie 14', 'Movie 15',
              'Movie 16', 'Movie 17', 'Movie 18', 'Movie 19', 'Movie 20'],
    'Review': ['Great movie!', 'Horrible film, do not watch.', 'It was okay, not the best.',
               'Amazing experience, highly recommend!', 'Terrible acting, poor storyline.',
               'The movie was a blast!', 'Worst movie ever, waste of time.',
               'Good storyline, great visuals.', 'Loved it, would watch again.',
               'Couldn’t finish it, so boring.',
               'Best movie of the year!', 'Very disappointing.',
               'Meh, didn’t live up to the hype.', 'Fantastic, loved every moment.',
               'Not worth the money.', 'Just okay, not great.',
               'Awesome acting, great plot.', 'A waste of time and money.',
               'Interesting story but slow-paced.', 'Boring and predictable.'],
    'Sentiment': ['Positive', 'Negative', 'Neutral', 'Positive', 'Negative',
                  'Positive', 'Negative', 'Positive', 'Positive', 'Negative',
                  'Positive', 'Negative', 'Neutral', 'Positive', 'Negative',
                  'Positive', 'Negative', 'Neutral', 'Negative', 'Positive']
}

# Preprocessing function
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return " ".join(words)

# Prepare the data
df = pd.DataFrame(data)
df['Review'] = df['Review'].apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['Review'])

# Convert sentiment to numerical values
df['Sentiment'] = df['Sentiment'].map({'Positive': 1, 'Negative': 0, 'Neutral': 2})

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, df['Sentiment'])

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    processed_review = preprocess_text(review)
    review_vec = vectorizer.transform([processed_review])
    prediction = model.predict(review_vec)

    sentiment_mapping = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
    sentiment = sentiment_mapping[prediction[0]]

    return render_template('result.html', review=review, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
