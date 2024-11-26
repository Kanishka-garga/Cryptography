fractionated_morse.py: Implementation algorithm of this cipher 
grandpe_cipher: Implemtation algorithm of this cipher

model_training.py
  Description: Script used to train the machine learning and deep learning models (Naive Bayes, Neural Network, LSTM, etc.).
  Purpose:
  ->Preprocesses the dataset.
  ->Trains models using labeled ciphertext data.
  ->Saves the trained models (.pkl, .h5) and preprocessing objects (e.g., TfidfVectorizer, LabelEncoder) for future use.
  
cipher_gui.py: Script for the graphical user interface (GUI) of the application.

naive_bayes_model.pkl
  Description: This file contains the pre-trained Naive Bayes model saved using joblib.
  Purpose: Used to predict cipher type based on features extracted from ciphertext
tfidf_vectorizer.pkl
  Description: A saved TfidfVectorizer instance used for converting input ciphertexts into numerical feature vectors.
  Purpose: Provides consistent transformation of ciphertexts into feature vectors for the models
label_encoder.pkl
  Description: A label encoder that maps cipher type labels to numeric values and vice versa.
  Purpose: Ensures consistent label mapping for training and predictions.
neural_network_model.h5
  Description: A pre-trained feedforward neural network model saved using the Keras library.
  Purpose: Predicts cipher type based on dense numerical representations of ciphertexts.
lstm_model.h5
  Description: A pre-trained Long Short-Term Memory (LSTM) model saved using the Keras library.
  Purpose: Predicts cipher type while considering sequential dependencies in the ciphertext.
