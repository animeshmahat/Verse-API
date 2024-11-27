from flask import Flask, request, jsonify
import pickle
import re
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

app = Flask(__name__)

stop_words = set(nltk.corpus.stopwords.words('english'))

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

# Modify sentence_similarity to incorporate TF-IDF
def sentence_similarity(sent1, sent2, tfidf_vectorizer=None):
    # Use TF-IDF for similarity calculation if available
    if tfidf_vectorizer:
        vector1 = tfidf_vectorizer.transform([sent1])
        vector2 = tfidf_vectorizer.transform([sent2])
        return cosine_similarity(vector1, vector2)[0][0]

    # Fallback to word overlap similarity
    words1 = tokenize_words(sent1)
    words2 = tokenize_words(sent2)
    all_words = list(set(words1 + words2))
    vector1 = [1 if word in words1 else 0 for word in all_words]
    vector2 = [1 if word in words2 else 0 for word in all_words]
    return cosine_similarity([vector1], [vector2])[0][0]

def rank_sentences(similarity_matrix, sentences, tfidf_vectorizer):
    if similarity_matrix.size == 0:
        return {}

    # Apply PageRank on the similarity matrix
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)

    # Advanced scoring: Add extra weights for key phrases, introductory, or concluding sentences
    for i, sentence in enumerate(sentences):
        if i == 0 or i == len(sentences) - 1:  # Introductory or concluding sentence
            scores[i] += 0.1
        if any(phrase in sentence.lower() for phrase in ["in conclusion", "to summarize"]):
            scores[i] += 0.2

    return scores

def remove_redundant_sentences(selected_sentences):
    seen = set()
    result = []
    for _, _, sentence in selected_sentences:
        if sentence not in seen:
            result.append(sentence)
            seen.add(sentence)
    return result

def generate_summary(sentences, sentence_scores, tfidf_vectorizer, summary_ratio=0.3, bullet_points=False):
    # Adaptive summary_ratio for shorter texts
    if len(sentences) < 10:
        summary_ratio = 0.5

    # Rank sentences and apply penalties for length
    ranked_sentences = sorted(
        ((i, sentence_scores[i] - (0.05 if len(s.split()) > 50 or len(s.split()) < 5 else 0), s)
         for i, s in enumerate(sentences)),
        key=lambda x: x[1],
        reverse=True
    )

    # Select top sentences based on summary_ratio
    summary_length = max(int(len(sentences) * summary_ratio), 1)
    selected_sentences = sorted(ranked_sentences[:summary_length], key=lambda x: x[0])

    # Remove redundant sentences
    final_sentences = remove_redundant_sentences(selected_sentences)

    if bullet_points:
        return [f"{s}" for s in final_sentences]
    else:
        summary = ' '.join(final_sentences)
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

        # Build similarity matrix and rank sentences using TF-IDF
        similarity_matrix = build_similarity_matrix(sentences)
        if similarity_matrix.size == 0:
            return jsonify({'paragraph': '', 'bullet_points': []}), 200

        # Rank sentences using advanced scoring
        sentence_scores = rank_sentences(similarity_matrix, sentences, vectorizer)

        # Generate summaries in both formats
        paragraph_summary = generate_summary(sentences, sentence_scores, vectorizer, summary_ratio=0.35)
        bullet_point_summary = generate_summary(sentences, sentence_scores, vectorizer, summary_ratio=0.35, bullet_points=True)

        return jsonify({
            'paragraph': paragraph_summary,
            'bullet_points': bullet_point_summary
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(threaded=True)
