# import pandas as pd 
# import random
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.preprocessing import LabelEncoder
# import joblib  # For saving/loading models
# import keras
# from keras import layers, models
# import numpy as np
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
# from datasets import Dataset

# # Load the combined dataset
# df = pd.read_csv('final_dataset.csv')
# X = df['Ciphertext']  # Features (Ciphertext)
# y = df['Algorithm']   # Labels (Algorithm)

# # Convert string labels to numeric labels using LabelEncoder
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)  # Apply label encoding

# # Split the dataset with stratification
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
# )

# # Vectorize the text using TF-IDF
# vectorizer = TfidfVectorizer(max_features=5000)
# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_test_tfidf = vectorizer.transform(X_test)

# # Results DataFrame
# results_df = pd.DataFrame({"Ciphertext": X_test, "True Algorithm": label_encoder.inverse_transform(y_test)})

# # ---------- Naive Bayes ----------
# def train_naive_bayes():
#     naive_bayes = MultinomialNB()
#     naive_bayes.fit(X_train_tfidf, y_train)
#     joblib.dump(naive_bayes, "naive_bayes_model.pkl")  # Save model
#     y_pred_nb = naive_bayes.predict(X_test_tfidf)
#     print("\nNaive Bayes Classification Report:")
#     print(classification_report(y_test, y_pred_nb))
#     print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
#     results_df["Naive Bayes Prediction"] = label_encoder.inverse_transform(y_pred_nb)

# # ---------- Transformer (BERT) ----------
# def train_transformer():
#     dataset = Dataset.from_dict({
#         'text': X_train.tolist(),
#         'label': y_train.tolist()
#     })

#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#     def tokenize_function(examples):
#         return tokenizer(examples['text'], padding=True, truncation=True)

#     encoded_dataset = dataset.map(tokenize_function, batched=True)
#     model = BertForSequenceClassification.from_pretrained(
#         'bert-base-uncased', num_labels=len(label_encoder.classes_)
#     )

#     training_args = TrainingArguments(
#         output_dir='./results',
#         evaluation_strategy="epoch",
#         per_device_train_batch_size=8,
#         per_device_eval_batch_size=8,
#         num_train_epochs=3,
#         logging_dir='./logs',
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=encoded_dataset,
#         eval_dataset=encoded_dataset
#     )
#     trainer.train()
#     model.save_pretrained("transformer_model_dir")  # Save model

#     # Prediction
#     test_encodings = tokenizer(list(X_test), truncation=True, padding=True, return_tensors="pt")
#     predictions = model(**test_encodings).logits.argmax(dim=1).numpy()
#     print("\nTransformer Classification Report:")
#     print(classification_report(y_test, predictions))
#     print("Transformer Accuracy:", accuracy_score(y_test, predictions))
#     results_df["Transformer Prediction"] = label_encoder.inverse_transform(predictions)

# # ---------- Neural Network ----------
# def train_neural_network():
#     X_train_dense = X_train_tfidf.toarray()
#     X_test_dense = X_test_tfidf.toarray()

#     model_nn = models.Sequential([
#         layers.Input(shape=(X_train_dense.shape[1],)),
#         layers.Dense(512, activation='relu'),
#         layers.Dropout(0.5),
#         layers.Dense(128, activation='relu'),
#         layers.Dense(len(label_encoder.classes_), activation='softmax')
#     ])
#     model_nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     model_nn.fit(X_train_dense, y_train, epochs=5, batch_size=64, validation_data=(X_test_dense, y_test))
#     model_nn.save("neural_network_model.h5")  # Save model

#     predictions = model_nn.predict(X_test_dense).argmax(axis=1)
#     print("\nNeural Network Classification Report:")
#     print(classification_report(y_test, predictions))
#     print("Neural Network Accuracy:", accuracy_score(y_test, predictions))
#     results_df["Neural Network Prediction"] = label_encoder.inverse_transform(predictions)

# # ---------- LSTM ----------
# def train_lstm():
#     X_train_dense = X_train_tfidf.toarray()
#     X_test_dense = X_test_tfidf.toarray()

#     X_train_reshaped = X_train_dense.reshape(X_train_dense.shape[0], 1, X_train_dense.shape[1])
#     X_test_reshaped = X_test_dense.reshape(X_test_dense.shape[0], 1, X_test_dense.shape[1])

#     model_lstm = models.Sequential([
#         layers.InputLayer(input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
#         layers.LSTM(128, return_sequences=True),
#         layers.LSTM(64),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(len(label_encoder.classes_), activation='softmax')
#     ])
#     model_lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     model_lstm.fit(X_train_reshaped, y_train, epochs=5, batch_size=64, validation_data=(X_test_reshaped, y_test))
#     model_lstm.save("lstm_model.h5")  # Save model

#     predictions = model_lstm.predict(X_test_reshaped).argmax(axis=1)
#     print("\nLSTM Classification Report:")
#     print(classification_report(y_test, predictions))
#     print("LSTM Model Accuracy:", accuracy_score(y_test, predictions))
#     results_df["LSTM Prediction"] = label_encoder.inverse_transform(predictions)

# if __name__ == "__main__":
#     print("Training Naive Bayes Model:")
#     train_naive_bayes()

#     print("\nTraining Transformer Model:")
#     train_transformer()

#     print("\nTraining Neural Network Model:")
#     train_neural_network()

#     print("\nTraining LSTM Model:")
#     train_lstm()

#     # Save prediction results
#     results_df.to_csv("evaluation_results.csv", index=False)
#     print("Prediction results saved to 'evaluation_results.csv'")

import pandas as pd  
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib  # For saving/loading models
import keras
from keras import layers, models
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Load the combined dataset
df = pd.read_csv('final_dataset.csv')
X = df['Ciphertext']  # Features (Ciphertext)
y = df['Algorithm']   # Labels (Algorithm)

# Convert string labels to numeric labels using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Apply label encoding

# Save the label encoder for later use
label_encoder_file = "label_encoder.pkl"
joblib.dump(label_encoder, label_encoder_file)
print(f"Label encoder saved as {label_encoder_file}")

# Split the dataset with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Save the TF-IDF vectorizer for later use
vectorizer_file = "tfidf_vectorizer.pkl"
joblib.dump(vectorizer, vectorizer_file)
print(f"TF-IDF vectorizer saved as {vectorizer_file}")

# Results DataFrame
results_df = pd.DataFrame({"Ciphertext": X_test, "True Algorithm": label_encoder.inverse_transform(y_test)})

# ---------- Naive Bayes ----------
def train_naive_bayes():
    naive_bayes = MultinomialNB()
    naive_bayes.fit(X_train_tfidf, y_train)
    joblib.dump(naive_bayes, "naive_bayes_model.pkl")  # Save model
    y_pred_nb = naive_bayes.predict(X_test_tfidf)
    print("\nNaive Bayes Classification Report:")
    print(classification_report(y_test, y_pred_nb))
    print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
    results_df["Naive Bayes Prediction"] = label_encoder.inverse_transform(y_pred_nb)

# ---------- Transformer (BERT) ----------
def train_transformer():
    dataset = Dataset.from_dict({
        'text': X_train.tolist(),
        'label': y_train.tolist()
    })

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding=True, truncation=True)

    encoded_dataset = dataset.map(tokenize_function, batched=True)
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=len(label_encoder.classes_)
    )

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset,
        eval_dataset=encoded_dataset
    )
    trainer.train()
    model.save_pretrained("transformer_model_dir")  # Save model

    # Prediction
    test_encodings = tokenizer(list(X_test), truncation=True, padding=True, return_tensors="pt")
    predictions = model(**test_encodings).logits.argmax(dim=1).numpy()
    print("\nTransformer Classification Report:")
    print(classification_report(y_test, predictions))
    print("Transformer Accuracy:", accuracy_score(y_test, predictions))
    results_df["Transformer Prediction"] = label_encoder.inverse_transform(predictions)

# ---------- Neural Network ----------
def train_neural_network():
    X_train_dense = X_train_tfidf.toarray()
    X_test_dense = X_test_tfidf.toarray()

    model_nn = models.Sequential([
        layers.Input(shape=(X_train_dense.shape[1],)),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])
    model_nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_nn.fit(X_train_dense, y_train, epochs=5, batch_size=64, validation_data=(X_test_dense, y_test))
    model_nn.save("neural_network_model.h5")  # Save model

    predictions = model_nn.predict(X_test_dense).argmax(axis=1)
    print("\nNeural Network Classification Report:")
    print(classification_report(y_test, predictions))
    print("Neural Network Accuracy:", accuracy_score(y_test, predictions))
    results_df["Neural Network Prediction"] = label_encoder.inverse_transform(predictions)

# ---------- LSTM ----------
def train_lstm():
    X_train_dense = X_train_tfidf.toarray()
    X_test_dense = X_test_tfidf.toarray()

    X_train_reshaped = X_train_dense.reshape(X_train_dense.shape[0], 1, X_train_dense.shape[1])
    X_test_reshaped = X_test_dense.reshape(X_test_dense.shape[0], 1, X_test_dense.shape[1])

    model_lstm = models.Sequential([
        layers.InputLayer(input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
        layers.LSTM(128, return_sequences=True),
        layers.LSTM(64),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])
    model_lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_lstm.fit(X_train_reshaped, y_train, epochs=5, batch_size=64, validation_data=(X_test_reshaped, y_test))
    model_lstm.save("lstm_model.h5")  # Save model

    predictions = model_lstm.predict(X_test_reshaped).argmax(axis=1)
    print("\nLSTM Classification Report:")
    print(classification_report(y_test, predictions))
    print("LSTM Model Accuracy:", accuracy_score(y_test, predictions))
    results_df["LSTM Prediction"] = label_encoder.inverse_transform(predictions)

if __name__ == "__main__":
    print("Training Naive Bayes Model:")
    train_naive_bayes()

    print("\nTraining Transformer Model:")
    train_transformer()

    print("\nTraining Neural Network Model:")
    train_neural_network()

    print("\nTraining LSTM Model:")
    train_lstm()

    # Save prediction results
    results_df.to_csv("evaluation_results.csv", index=False)
    print("Prediction results saved to 'evaluation_results.csv'")
