from flask import Flask, request, jsonify
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Initialize Flask app
app = Flask(__name__)

# Load the logistic regression model and TF-IDF vectorizer from pickle files
with open('logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)


# Preprocessing function (adjust according to your notebook's logic)
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove URLs
    text = re.sub('https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub('<.*?>+', '', text)

    # Remove punctuation
    text = re.sub('[%s]' % re.escape('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'), '', text)

    # Remove numbers
    text = re.sub('\w*\d\w*', '', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]

    # Rejoin tokens into a single string
    processed_text = ' '.join(tokens)

    return processed_text


# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Preprocess the text
        processed_text = preprocess_text(text)

        # Transform the text using the TF-IDF vectorizer
        text_vectorized = vectorizer.transform([processed_text])

        # Predict sentiment using the logistic regression model
        prediction = model.predict(text_vectorized)

        # Return the prediction as JSON
        response = {'sentiment': prediction[0]}  # Assuming sentiment is binary (0 or 1)
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
