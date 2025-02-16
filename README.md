# YouTube-Bot-NN-Classifier

## 📌 About This Project

This project trains a Neural Network (NN) Classifier using PyTorch and Term Frequency-Inverse Document Frequency (TF-IDF) vectorization to analyze YouTube comments and determine whether they are malicious bot comments or human-written.
Read more about my project @ https://daniwave100.github.io/DanielPersonalPage/ !

## 🔍 Features

- ✔ Text Preprocessing – Removes stopwords, applies stemming, and vectorizes text using TF-IDF
- ✔ Neural Network Model – Built using PyTorch with multiple hidden layers for classification
- ✔ Binary Classification (Human vs. Bot) – Predicts if a comment is Human (0) or Bot (1)

## 🛠️ Tech Stack
	-	Python – Core programming language
	-	PyTorch – Deep learning framework for the neural network
	-	NLTK – Natural Language Processing (NLP) for text preprocessing
	-	scikit-learn – TF-IDF vectorization and dataset splitting
	-	Pandas & NumPy – Data handling and manipulation

## 📊 Model Training & Evaluation
	-Training Data: Web-scraped YouTube comments
	-Feature Extraction: TF-IDF applied to comment corpus
	-Model Architecture:
	  -- Input Layer: TF-IDF vectorized comment features
	  -- Hidden Layers: Fully connected layers with ReLU activation
	  -- Output Layer: LogSoftmax activation for classification
	  -- Loss Function: Negative Log-Likelihood Loss (NLLLoss)
	  -- Optimizer: Adam optimizer to update weights and minimize loss

 ## 📈 Future Improvements
- ✅ Expand training dataset for improved accuracy (HIGH PRIORITY)
- ✅ Implement transformer-based models for better performance
- ✅ Deploy model with a Flask API for real-time predictions and user-friendly analysis

## ⚠️ Current Development Status

🔬 This project is in its preliminary stages and currently serves as a simple prototype for detecting AI-generated comments. The model is based on a basic feedforward neural network with TF-IDF features and has limited accuracy due to a relatively small dataset. 

💡 The goal is to improve accuracy and performance for classification accuracy of 90% or more. In the midst of this, I hope to also develop my understanding of neural networks, natural language processing, and overall, AI.

