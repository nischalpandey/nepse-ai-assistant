import re
import numpy as np

from transformers import pipeline, AutoTokenizer, AutoModel # used for question-answering
from sentence_transformers import SentenceTransformer # used for semantic similarity
import logging # used for logging
import yaml
import json
import pickle # used for saving and loading the model
from sklearn.feature_extraction.text import TfidfVectorizer # used for vectorizing text data
from sklearn.metrics.pairwise import cosine_similarity # used for calculating similarity between vectors
from nltk.tokenize import word_tokenize # used for preprocessing text
from nltk.corpus import stopwords # used for preprocessing text
from nltk.stem import WordNetLemmatizer # used for preprocessing text
import torch # used for loading the model and making predictions 



#importing the intents from the utils folder
from nepsebot.ai.utils.intent import getintents, user_data, get_stock_price, get_stock_history
from nepsebot.ai.da.predict import getpredictchart

from nepsebot.ai.da.allda import generate_da
# Setup logging
logging.basicConfig(level=logging.INFO)

# Load PreProcessor Natural Language Processing (NLP) model
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')




# Regex patterns for intents
intent_patterns = getintents()
class ConversationModel:
    def __init__(self):
        self.categories = None
        self.conversations = None
        self.vectorizer = TfidfVectorizer()
        self.conversation_vectors = None
        self.sentence_transformer = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def load_data(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as file:
            if data_file.endswith('.yml') or data_file.endswith('.yaml'):
                data = yaml.safe_load(file)
            elif data_file.endswith('.json'):
                data = json.load(file)
            else:
                raise ValueError("Unsupported file format. Use .yml, .yaml, or .json")
            self.categories = data['categories']
            self.conversations = data['conversations']


        
 
    @classmethod
    def load_model(cls, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls()
        model.categories = data['categories']
        model.conversations = data['conversations']
        model.vectorizer = data['vectorizer']
        model.conversation_vectors = data['conversation_vectors']
        model.sentence_embeddings = data['sentence_embeddings']
        return model

    def get_response(self, user_input):
        preprocessed_input = self.preprocess_text(user_input)
        user_vector = self.vectorizer.transform([preprocessed_input])
        tfidf_similarities = cosine_similarity(user_vector, self.conversation_vectors).flatten()

        user_embedding = self.sentence_transformer.encode([preprocessed_input])
        semantic_similarities = cosine_similarity(user_embedding, self.sentence_embeddings).flatten()

        combined_similarities = (tfidf_similarities + semantic_similarities) / 2
        most_similar_idx = combined_similarities.argmax()

        return self.conversations[most_similar_idx][1]

   
def get_most_similar_query(query, predefined_queries, model):
    query_embedding = model.sentence_transformer.encode([query])
    predefined_embeddings = model.sentence_transformer.encode(predefined_queries)
    
    similarities = cosine_similarity(query_embedding, predefined_embeddings).flatten()
    best_match_index = similarities.argmax()
    
    return predefined_queries[best_match_index]


def process_user_query(user_input, conversation_model):
    """
    Process the user query and return an appropriate response

    Args:
    user_input (str): The user query
    conversation_model (ConversationModel): The conversation model to use for responses
    
    """

    # We use intent because we have specific tasks to perform based on user input
    # It can work as customer support bot aslo
    for intent, pattern in intent_patterns.items():
        if pattern.search(user_input):
            if intent == "stock_price":
                symbol_match = re.search(r'\b[A-Z]+\b', user_input)
                if symbol_match:
                    symbol = symbol_match.group(0)
                    price = get_stock_price(symbol)
                    if price:
                        return f"Let me check that for you. The current price of {symbol} is NPR {price:.2f}"
                    else:
                        return f"Sorry, I couldn't fetch the price for {symbol}"
            elif intent == "stock_history":
                symbol_match = re.search(r'\b[A-Z]+\b', user_input)
                if symbol_match:
                    symbol = symbol_match.group(0)
                    history = get_stock_history(symbol)
                    if history:
                        return f"Here's the 30-day price history for {symbol}: {history}"
                    else:
                        return f"Sorry, I couldn't fetch the history for {symbol}"
            elif intent == "predict_stock":
                symbol_match = re.search(r'\b[A-Z]+\b', user_input)
                if symbol_match:
                    symbol = symbol_match.group(0)
                    try:
                        response = getpredictchart()
                        return response

                    except Exception as e:
                        logging.error(f"Error predicting stock price: {e}")
                       

                    return f"Let me check that for you. I'm sorry, I don't have the capability to predict stock prices at the moment."
            elif intent == "analyze_stock":
                response = generate_da()
                if response:
                    return response
                return "I'm sorry, I don't have the resources to analyze stocks at the moment."
            elif intent == "user_email":
                return f"Your email is {user_data['email']}"
            elif intent == "user_name":
                return f"Your name is {user_data['name']}"
            elif intent == "user_account":
                return f"Your account number is {user_data['account_number']}"
            elif intent == "about_stock":
                return "I'm sorry, I don't have information about that stock at the moment."
    
    #  we can use first QA model from huggingface transformers library
    # to get the answer for the user query 
    # We are not using that model , The model is large and have unnecessary data
    # So we are using our own model to get the answers
    
    #->More Training data can be added to the model to get better results 
    
    #-> It was trained on very  few data so the results may not be accurate

    response = conversation_model.get_response(user_input)
    
    if response:
        return response
    # If no response is found, we can use semantic similarity to find the most similar query
    predefined_queries = [conv[0] for conv in conversation_model.conversations]
    similar_query = get_most_similar_query(user_input, predefined_queries, conversation_model)
    response1 = conversation_model.get_response(similar_query)
    if response1:
        return response1
    return "Please ask a question related to NEPSE."

