#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 16:52:49 2023

@author: wcchun
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import PyPDF2
import re
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Define the path to save uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Limit upload file size
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Allowed extensions for file upload
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
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

def extract_text_from_pdf(pdf_path):
    """Extract text from the provided PDF file."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ' '.join([page.extract_text() for page in reader.pages])
    return text

def preprocess_text(text):
    """Preprocess the text: remove URLs, tokenize, remove stopwords, and apply lemmatization."""
    text = re.sub(r'https?:\/\/\S+', '', text)
    vectorizer = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS), max_features=500, ngram_range=(1, 2))
    tokens = vectorizer.fit_transform([text])
    cleaned_text = ' '.join(vectorizer.get_feature_names_out())
    return cleaned_text, vectorizer

def save_plot(topics, topic_weights):
    """Save the topic weights as a bar plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    y_pos = range(len(topics))
    ax.barh(y_pos, topic_weights, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(topics)
    ax.invert_yaxis()
    plt.savefig("static/plot.png", bbox_inches='tight')

def extract_and_analyze(pdf_path):
    """Extract text from PDF, preprocess it, and analyze topics."""
    extracted_text = extract_text_from_pdf(pdf_path)
    preprocessed_text, vectorizer = preprocess_text(extracted_text)
    
    # Apply NMF for topic modeling
    num_topics = 5
    nmf = NMF(n_components=num_topics, random_state=42).fit(vectorizer.transform([preprocessed_text]))
    
    # Get topics and topic weights
    topics = []
    topic_weights = []
    for topic_idx, topic in enumerate(nmf.components_):
        topic_words = [vectorizer.get_feature_names_out()[i] for i in sorted(range(len(topic)), key=lambda i: topic[i])[-5:]]
        topics.extend(topic_words)
        weights = sorted(topic)[-5:]
        topic_weights.extend(weights)

    # Sort topics based on weights in descending order
    topics = [x for _, x in sorted(zip(topic_weights, topics), reverse=True)]
    topic_weights = sorted(topic_weights, reverse=True)
    
    # Save the plot
    save_plot(topics, topic_weights)
    
    return topics[:5]

if __name__ == '__main__':
    app.run(debug=True)







