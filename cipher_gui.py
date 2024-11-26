import tkinter as tk
from tkinter import messagebox
import joblib
from keras import layers, models
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertForSequenceClassification
import torch

naive_bayes_model = joblib.load("naive_bayes_model.pkl")  # Naive Bayes Model
vectorizer = joblib.load("tfidf_vectorizer.pkl")          # TfidfVectorizer
label_encoder = joblib.load("label_encoder.pkl")          # Label Encoder

# Loaded neural network and LSTM models
neural_network_model = models.load_model("neural_network_model.h5")
lstm_model = models.load_model("lstm_model.h5")

# Loaded BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
transformer_model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', 
    num_labels=len(label_encoder.classes_)
)

def predict_cipher(ciphertext):
    try:
        vectorized_input = vectorizer.transform([ciphertext])
        dense_input = vectorized_input.toarray()
        reshaped_input = dense_input.reshape(dense_input.shape[0], 1, dense_input.shape[1])  # For LSTM

        # Naive Bayes Prediction and Confidence Score
        nb_probs = naive_bayes_model.predict_proba(vectorized_input)[0]
        nb_prediction = label_encoder.inverse_transform([nb_probs.argmax()])[0]
        nb_confidence = nb_probs.max()

        # Neural Network Prediction and Confidence Score
        nn_probs = neural_network_model.predict(dense_input, verbose=0)[0]
        nn_prediction = label_encoder.inverse_transform([nn_probs.argmax()])[0]
        nn_confidence = nn_probs.max()

        # LSTM Prediction and Confidence Score
        lstm_probs = lstm_model.predict(reshaped_input, verbose=0)[0]
        lstm_prediction = label_encoder.inverse_transform([lstm_probs.argmax()])[0]
        lstm_confidence = lstm_probs.max()

        # Transformer Prediction and Confidence Score
        inputs = tokenizer(ciphertext, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = transformer_model(**inputs)
        transformer_probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
        transformer_prediction = label_encoder.inverse_transform([transformer_probs.argmax()])[0]
        transformer_confidence = transformer_probs.max()

        # Stored all predictions and confidences for GUI
        results = {
            'Naive Bayes': (nb_prediction, nb_confidence),
            'Neural Network': (nn_prediction, nn_confidence),
            'LSTM': (lstm_prediction, lstm_confidence),
            'Transformer': (transformer_prediction, transformer_confidence),
        }

        result_message = (
            f"Naive Bayes Prediction: {nb_prediction}\nConfidence: {nb_confidence:.2f}\n\n"
            f"Neural Network Prediction: {nn_prediction}\nConfidence: {nn_confidence:.2f}\n\n"
            f"LSTM Prediction: {lstm_prediction}\nConfidence: {lstm_confidence:.2f}\n\n"
            f"Transformer Prediction: {transformer_prediction}\nConfidence: {transformer_confidence:.2f}"
        )
        messagebox.showinfo("Prediction Results", result_message)

        return results

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


def show_model_prediction(model_name, predictions):
    if predictions is None:
        messagebox.showerror("Error", "Please predict first before selecting a model.")
        return

    prediction, confidence = predictions[model_name]
    messagebox.showinfo(
        f"{model_name} Prediction",
        f"Prediction from {model_name} model: {prediction}\nConfidence: {confidence:.2f}"
    )

def create_gui():
    root = tk.Tk()
    root.title("Cipher Type Predictor")
    
    tk.Label(root, text="Enter Ciphertext:").grid(row=0, column=0, padx=10, pady=10)
    ciphertext_entry = tk.Entry(root, width=50)
    ciphertext_entry.grid(row=0, column=1, padx=10, pady=10)

    predictions = None

    def on_predict():
        nonlocal predictions
        ciphertext = ciphertext_entry.get()
        predictions = predict_cipher(ciphertext)

    tk.Button(root, text="Show All Model Predictions", command=on_predict).grid(row=1, column=0, columnspan=2, pady=20)

    tk.Button(root, text="Show Naive Bayes Prediction", command=lambda: show_model_prediction('Naive Bayes', predictions)).grid(row=2, column=0, padx=10, pady=10)
    tk.Button(root, text="Show Neural Network Prediction", command=lambda: show_model_prediction('Neural Network', predictions)).grid(row=2, column=1, padx=10, pady=10)
    tk.Button(root, text="Show LSTM Prediction", command=lambda: show_model_prediction('LSTM', predictions)).grid(row=3, column=0, padx=10, pady=10)
    tk.Button(root, text="Show Transformer Prediction", command=lambda: show_model_prediction('Transformer', predictions)).grid(row=3, column=1, padx=10, pady=10)

    root.mainloop()


if __name__ == "__main__":
    create_gui()
