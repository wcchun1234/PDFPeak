#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 16:52:49 2023

@author: wcchun
"""

# Flask setup - https://flask.palletsprojects.com/en/2.2.x/patterns/fileuploads/
# standard setup for a Flask application
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Define the path to save uploaded files - https://flask.palletsprojects.com/en/2.2.x/patterns/fileuploads/
# Standard practice in Flask applications
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Limit upload file size to 16 MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Allowed extensions for file upload
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    # Check for allowed file extensions - https://flask.palletsprojects.com/en/2.2.x/patterns/fileuploads/
    # Typical function in Flask file uploads
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    # Render the index page - https://flask.palletsprojects.com/en/2.2.x/patterns/fileuploads/
    # Standard Flask usage
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Handle file upload - https://flask.palletsprojects.com/en/2.2.x/patterns/fileuploads/
    # Common pattern in Flask, could be adapted from documentation
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        topics = extract_and_analyze(filepath)
        return render_template('results.html', topics=topics)
    return redirect(request.url)

# PDF text extraction using PyPDF2 - https://pypdf2.readthedocs.io/en/latest/
import PyPDF2
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ' '.join([page.extract_text() for page in reader.pages])
    return text

# Preprocessing text for NMF - https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
def preprocess_text(text):
    text = re.sub(r'https?:\/\/\S+', '', text)  # URL removal
    # TF-IDF vectorization - https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
    # scikit-learn examples
    vectorizer = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS), max_features=500, ngram_range=(1, 2))
    tokens = vectorizer.fit_transform([text])
    cleaned_text = ' '.join(vectorizer.get_feature_names_out())
    return cleaned_text, vectorizer

# Plotting function using matplotlib - https://matplotlib.org/stable/users/explain/quick_start.html#a-simple-example
# from matplotlib examples
import matplotlib.pyplot as plt
def save_plot(topics, topic_weights):
    fig, ax = plt.subplots(figsize=(12, 6))
    y_pos = range(len(topics))
    ax.barh(y_pos, topic_weights, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(topics)
    ax.invert_yaxis()
    plt.savefig("static/plot.png", bbox_inches='tight')

# NMF for topic extraction - https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
# from scikit-learn documentation/examples
from sklearn.decomposition import NMF
def extract_and_analyze(pdf_path):
    extracted_text = extract_text_from_pdf(pdf_path)
    preprocessed_text, vectorizer = preprocess_text(extracted_text)
    
    # Apply NMF for topic modeling
    num_topics = 5
    nmf = NMF(n_components=num_topics, random_state=42).fit(vectorizer.transform([preprocessed_text]))
    
    # Extract and process topics and their weights
    topics = []
    topic_weights = []
    for topic_idx, topic in enumerate(nmf.components_):
        topic_words = [vectorizer.get_feature_names_out()[i] for i in sorted(range(len(topic)), key=lambda i: topic[i])[-5:]]
        topics.extend(topic_words)
        weights = sorted(topic)[-5:]
        topic_weights.extend(weights)

    # Sort topics and weights for visualization
    topics = [x for _, x in sorted(zip(topic_weights, topics), reverse=True)]
    topic_weights = sorted(topic_weights, reverse=True)
    
    # Save plot of topics and weights
    save_plot(topics, topic_weights)
    
    return topics[:5]

# Run the Flask app - https://flask.palletsprojects.com/en/2.2.x/patterns/fileuploads/
if __name__ == '__main__':
    app.run(debug=True)
