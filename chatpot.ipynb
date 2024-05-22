import json
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

# Load data from the JSON file
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Create a dictionary from the intents
data = {}
for intent in intents['intents']:
    for pattern in intent['patterns']:
        data[pattern] = random.choice(intent['responses'])

# Preprocess text
def preprocess(text):
    # Tokenization
    tokens = word_tokenize(text.lower())

    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Preprocess and vectorize the data
preprocessed_data = [preprocess(question) for question in data.keys()]
vectorized_data = tfidf_vectorizer.fit_transform(preprocessed_data)

# Main loop
print("Chatbot: Hello! How can I help you?")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'goodbye':
        print("Chatbot: Goodbye! Have a nice day!")
        break

    # Preprocess user input
    preprocessed_input = preprocess(user_input)

    # Vectorize user input
    vectorized_input = tfidf_vectorizer.transform([preprocessed_input])

    # Find the most similar question
    similarities = cosine_similarity(vectorized_input, vectorized_data)
    most_similar_index = similarities.argmax()
    most_similar_question = list(data.keys())[most_similar_index]

    # If similarity is above a threshold, consider it a match
    if similarities[0, most_similar_index] > 0.5:
        response = data[most_similar_question]
    else:
        response = "Sorry, I don't understand. Can you ask another question?"

    print("Chatbot:", response)