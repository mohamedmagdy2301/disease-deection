# Basic Libraries
from pyexpat import model
import numpy as np
import pandas as pd
import sklearn

# Necessary Libraries for Data Preparation
import string
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Necessary Libraries for ML Models
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Necessary Libraries for Accuracy Measures
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Necessary Libraries for Deployment
import joblib
from flask import Flask, jsonify, request,app

from flask_cors import CORS  

app = Flask(__name__)
CORS(app)

# download the Punkt tokenizer models
nltk.download('punkt')

# download a list of common stopwords
nltk.download('stopwords')

# download the WordNet lexical database
nltk.download('wordnet')
data = pd.read_csv("/Users/mohamedmagdy/Desktop/manar/final_flutter_app/flask_and_model/Symptom2Disease.csv")
data
# Drop the 'Unnamed: 0' column
data.drop(columns = ["Unnamed: 0"], inplace = True)
data
# Concise summary of the DataFrame's structure and content
data.info()
data.columns
data.shape
# Count the number of unique values in each column
data.nunique()
data.value_counts().sum()
# Check and Count null values
data.isnull().sum()
# Check and Count duplicated values
data.duplicated().sum()
# Drop duplicated values
data.drop_duplicates(inplace = True)
data
def lowercase_text(text):
    return text.lower()

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    text_without_punct = text.translate(translator).strip()
    return text_without_punct

def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return filtered_tokens

def lemmatize_text(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

# Preprocessing Container function

def preprocess_text(text):
    text = lowercase_text(text)
    text = remove_punctuation(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_text(tokens)
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Apply Preprocessing Container function to symptoms

data['text'] = data['text'].apply(preprocess_text)

# Extract and Count unique dictionary vocabs

def count_unique_vocab(count):
    unique_vocabularies = set()
    for text in count:
        words = text.split()
        for word in words:
            unique_vocabularies.add(word)
    return len(unique_vocabularies)

# Count unique dictionary vocabs
num_unique_vocabs = count_unique_vocab(data['text'])

print("Number of unique dictionary vocabs:", num_unique_vocabs)


X = data['text']
y = data['label']

X
y
# The 'shuffle' function is used to randomly Shuffle/Rearrange the elements of a dataset
data = shuffle(data, random_state = 42)
data
# Charactieristics of the data
info = data.describe().round()
info
# from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

# Text feature extraction using TF-IDF vectorizer to transform text data
tfidf_vectorizer = TfidfVectorizer(max_features=2400)

# Transforming training and testing data
X_train = tfidf_vectorizer.fit_transform(X_train).toarray()
X_test = tfidf_vectorizer.transform(X_test).toarray()

def tfidf_vectorize_text(text_data, max_features=2400):
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(text_data)
    return tfidf_matrix


# Create a Decision Tree Classifier object
dt_classifier = DecisionTreeClassifier(random_state = 42)

# Define the hyperparameters and their possible values to search
parameters = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features' : ['sqrt', 'log2', None]
}

# Create the Grid Search object
grid_search = GridSearchCV(estimator = dt_classifier,
                           param_grid = parameters,
                           cv = 5,
                           scoring = 'accuracy',
                           n_jobs = -1)

# Fit the Grid Search to the train data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters found
best_hyperparameters = grid_search.best_params_
print("Best Hyperparameters:", best_hyperparameters)

# Get the best model version
best_dt_classifier = grid_search.best_estimator_
print(best_dt_classifier)

# Print the best accuracy found
best_accuracy = grid_search.best_score_
print(f'Best Accuracy: {best_accuracy*100:.2f} %')

# Calculate and Compare the Score of train data and test data

train_score = best_dt_classifier.score(X_train, y_train)
test_score = best_dt_classifier.score(X_test, y_test)

# Print the scores
print(f'Training Score: {train_score*100:.2f} %')
print(f'Testing Score: {test_score*100:.2f} %')





# Make Predictions on the train data and test data

train_predictions = best_dt_classifier.predict(X_train)
test_predictions = best_dt_classifier.predict(X_test)



# Calculate and Compare the Accuracy for training and testing data
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

# Print the accuracies
print(f'Training Accuracy: {train_accuracy*100:.2f} %')
print(f'Testing Accuracy: {test_accuracy*100:.2f} %')



# Make the Confusion Matrix
cm_1 = confusion_matrix(y_test, test_predictions)

# Print the Confusion Matrix

# print(cm_1)

# Validation Test 

#text_before = "The skin around my mouth, nose, and eyes is ruddy and kindled. It is regularly bothersome and awkward. There's a recognizable aggravation in my nails."
# text_before = "The abdominal pain has been coming and going, and it's been really unpleasant. It's been accompanied by constipation and vomiting. I feel really concerned about my health."

#-------------------end ml model ----------------------------------

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Extract features from the JSON data from user
    text_before = data['text']
    
    # Cleaning
    text_after = preprocess_text(text_before)
    print(text_before)
    print(text_after)

   
    # Vectorization
    tfidf_vectorizer

    text_after = tfidf_vectorizer.transform([text_after]).toarray()
    # Make prediction
    prediction =  best_dt_classifier.predict(text_after)
    print(prediction[0])
    return jsonify({'prediction': prediction[0]})   

if __name__ == '__main__':
    app.run(debug=True)