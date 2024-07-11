import json
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle
from keras.models import load_model
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import datetime
import os
import streamlit as st

# Set the NLTK data path
nltk.data.path.append(os.path.join(os.path.expanduser('~'), 'nltk_data'))

# Download necessary NLTK data
nltk.download('punkt', force=True)
nltk.download('wordnet', force=True)
nltk.download('omw-1.4', force=True)
nltk.download('vader_lexicon', force=True)

# Clear NLTK cache
from nltk.data import clear_cache
clear_cache()

# Check if the JSON file exists and is not empty
file_path = 'intents.json'
if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
    with open(file_path, 'r') as file:
        data = json.load(file)
else:
    raise FileNotFoundError(f"The file {file_path} does not exist or is empty.")

# Preprocess the data
lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore = ['?', '!', ',', '.']

for intent in data['intents']:
    for pattern in intent['patterns']:
        words_list = nltk.word_tokenize(pattern)
        words.extend(words_list)
        documents.append((words_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in pattern_words]
    for w in words:
        bag.append(1 if w in pattern_words else 0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append(bag + output_row)

np.random.shuffle(training)
training = np.array(training)
x_train = training[:, :len(words)]
y_train = training[:, len(words):]

# Model building
model = Sequential()
model.add(Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

adam = Adam(learning_rate=0.001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(np.array(x_train), np.array(y_train), epochs=200, batch_size=10, verbose=1)

# Save the model in the native Keras format
model.save('chatbot_model.keras')

# Load the model
model = load_model('chatbot_model.keras')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def clean_up(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    threshold = 0.8
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for result in results:
        return_list.append({'intent': classes[result[0]], 'probability': str(result[1])})
    return return_list

def get_response(return_list, data_json):
    if len(return_list) == 0:
        tag = 'no-answer'
    else:
        tag = return_list[0]['intent']
    
    if tag == 'datetime':
        current_time = datetime.datetime.now().strftime("%A, %d %B %Y, %H:%M:%S")
        return current_time
    
    intent_list = data_json['intents']
    for intent in intent_list:
        if tag == intent['tag']:
            result = np.random.choice(intent['responses'])
    return result

def response(text):
    return_list = predict_class(text)
    response = get_response(return_list, data)
    return response

# Ensure spaCy model is downloaded
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

sid = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    scores = sid.polarity_scores(text)
    sentiment = "positive" if scores["compound"] >= 0 else "negative"
    return sentiment

def recognize_entities(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities

def generate_response(sentiment, entities, intents_json):
    if sentiment == "positive" and "greeting" in entities:
        response = "Hello! How can I assist you today?"
    else:
        response = "I'm sorry to hear that. How can I help you feel better?"
    return response

def main():
    st.title("Advanced Chatbot with Sentiment Analysis and Entity Recognition")
    
    input_text = st.text_input("You:")
    
    if st.button("Send"):
        if input_text.strip() != "":
            sentiment = analyze_sentiment(input_text)
            entities = recognize_entities(input_text)
            bot_response = generate_response(sentiment, entities, data)
            st.write(f"Chatbot: {bot_response}")

if __name__ == "__main__":
    main()
