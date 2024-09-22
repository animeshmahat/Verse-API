from flask import Flask, request, jsonify
import pickle
import re
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Initialize Flask app
app = Flask(__name__)

# Pre-load NLTK resources at app startup
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

# Load the logistic regression model and TF-IDF vectorizer from pickle files
with open('logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Preprocessing function for sentiment analysis
def preprocess_text(text):
    text = text.lower()
    text = re.sub('https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub('<.*?>+', '', text)  # Remove HTML tags
    text = re.sub('[%s]' % re.escape('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'), '', text)  # Remove punctuation
    text = re.sub('\w*\d\w*', '', text)  # Remove numbers
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Define the prediction route for sentiment analysis
@app.route('/predict', methods=['POST'])
def predict():
    try:
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

        response = {'sentiment': prediction[0]}
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Define preprocessing and summarization functions for TextRank
def preprocess_text_for_summarization(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\d+\.\s*', '', text)
    text = re.sub(r'•\s*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_sentences(text):
    return sent_tokenize(text)

# Use cached stopwords and filter out non-alphanumeric tokens
def tokenize_words(sentence):
    words = word_tokenize(sentence.lower())
    return [word for word in words if word not in stop_words and word.isalnum()]

def sentence_similarity(sent1, sent2):
    words1 = tokenize_words(sent1)
    words2 = tokenize_words(sent2)
    all_words = list(set(words1 + words2))
    vector1 = [1 if word in words1 else 0 for word in all_words]
    vector2 = [1 if word in words2 else 0 for word in all_words]
    return cosine_similarity([vector1], [vector2])[0][0]

# Build similarity matrix
def build_similarity_matrix(sentences):
    if len(sentences) == 0:
        return np.array([])
    matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                matrix[i][j] = sentence_similarity(sentences[i], sentences[j])
    return matrix

def rank_sentences(similarity_matrix):
    if similarity_matrix.size == 0:
        return {}
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)
    return scores

# Summarize function with no logic change
def generate_summary(sentences, sentence_scores, summary_ratio=0.3, min_sentence_length=5, bullet_points=False):
    # Rank sentences by score but keep track of their original index
    ranked_sentences = sorted(((i, sentence_scores[i], s) for i, s in enumerate(sentences)), key=lambda x: x[1],
                              reverse=True)
    ranked_sentences = [(i, score, s) for i, score, s in ranked_sentences if len(s.split()) >= min_sentence_length]

    # Select the top sentences based on summary_ratio
    summary_length = max(int(len(sentences) * summary_ratio), 1)
    selected_sentences = sorted(ranked_sentences[:summary_length], key=lambda x: x[0])  # Sort by the original index

    if bullet_points:
        return [f"{s}" for _, _, s in selected_sentences]  # Return as bullet points in original order
    else:
        summary = ' '.join([s for _, _, s in selected_sentences])
        if not summary.endswith(('.', '?', '!')):
            summary += '.'
        return summary

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()
        title = data.get('title', '')
        description = data.get('description', '')

        if not title and not description:
            return jsonify({'error': 'No title or description provided'}), 400

        # Combine title and description into a single text
        text = f"{title}. {description}" if title else description

        # Preprocess the text for summarization
        clean_text = preprocess_text_for_summarization(text)

        # Tokenize into sentences
        sentences = tokenize_sentences(clean_text)

        # Build similarity matrix and rank sentences
        similarity_matrix = build_similarity_matrix(sentences)
        if similarity_matrix.size == 0:
            return jsonify({'paragraph': '', 'bullet_points': []}), 200
        sentence_scores = rank_sentences(similarity_matrix)

        # Generate summary in both formats
        paragraph_summary = generate_summary(sentences, sentence_scores, summary_ratio=0.35)
        bullet_point_summary = generate_summary(sentences, sentence_scores, summary_ratio=0.35, bullet_points=True)

        return jsonify({
            'paragraph': paragraph_summary,
            'bullet_points': bullet_point_summary
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(threaded=True)
