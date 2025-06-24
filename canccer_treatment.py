pip install googletrans

from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from googletrans import Translator
import os

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained NLP model for AI responses (text-generation)
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model_gpt = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
chatbot = pipeline("text-generation", model=model_gpt, tokenizer=tokenizer)

# Initialize the translator
translator = Translator()

# Model training and saving
def train_and_save_model():
    # Load sample data (Iris dataset for this example)
    data = load_iris()
    X = data.data
    y = data.target

    # One-hot encode the target variable
    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y.reshape(-1, 1))

    # Standardize the feature data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define a simple Sequential model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(y_train.shape[1], activation='softmax')  # Number of classes in the output layer
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=8, validation_split=0.1)

    # Save the trained model as an .h5 file
    model.save('my_cancer_treatment_model.h5')

    print("Model trained and saved as 'my_cancer_treatment_model.h5'")

    return model

# Load or train the model
model_path = 'my_cancer_treatment_model.h5'

def load_or_train_model(model_path):
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully.")
        except ValueError as e:
            print(f"Model loading error: {e}")
            print("Training a new model...")
            model = train_and_save_model()
    else:
        print("Model file not found. Training a new model...")
        model = train_and_save_model()

    return model

# Load or define your ML model for cancer treatment suggestion
model = load_or_train_model(model_path)

# Preprocessing function for the ML model input
def preprocess_input(data):
    scaler = StandardScaler()
    processed_data = scaler.fit_transform(np.array(list(data.values())).reshape(1, -1))
    return processed_data

# Chatbot response function
def get_chatbot_response(user_input, target_language):
    response = chatbot(user_input, max_length=50, do_sample=True, top_k=10)[0]['generated_text']
    translated_response = translator.translate(response, dest=target_language)
    return translated_response.text

# ML model prediction function
def predict_treatment(data):
    processed_data = preprocess_input(data)
    prediction = model.predict(processed_data)
    return prediction

# API route for interacting with the chatbot
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    user_data = request.json.get('data')
    source_language = request.json.get('language', 'en')

    # Translate user input to English
    translated_input = translator.translate(user_input, dest='en').text

    # Get AI response
    ai_response = get_chatbot_response(translated_input, source_language)

    # If user data is provided, make a prediction
    if user_data:
        ml_prediction = predict_treatment(user_data)
        return jsonify({
            'response': ai_response,
            'prediction': ml_prediction.tolist()
        })

    # Otherwise, just return the AI response
    return jsonify({'response': ai_response})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
