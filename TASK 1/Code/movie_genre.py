import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import randint
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Function to preprocess text (remove punctuation, stopwords, and lemmatization)
def preprocess_text(text):
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    words = nltk.word_tokenize(text)
    # Remove stopwords and apply lemmatization
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Load training data (no GENRE column in test_data for predictions)
train_data = pd.read_csv("/Users/satwikreddy/Downloads/Genre Classification Dataset/train_data.txt", sep=":::", header=None, names=["ID", "TITLE", "GENRE", "DESCRIPTION"], engine='python')
test_data = pd.read_csv("/Users/satwikreddy/Downloads/Genre Classification Dataset/test_data.txt", sep=":::", header=None, names=["ID", "TITLE", "DESCRIPTION"], engine='python')

# Data Preprocessing
X_train = train_data['DESCRIPTION'].apply(preprocess_text)
y_train = train_data['GENRE']
X_test = test_data['DESCRIPTION'].apply(preprocess_text)  # We will only use descriptions for predictions

# Check class distribution in training data
print("Class distribution in training data:")
print(y_train.value_counts())

# TF-IDF Vectorization with trigrams (for better feature extraction)
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 3), sublinear_tf=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model Initialization with class_weight='balanced' to handle class imbalance
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')

# Hyperparameter distribution for RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 150, 200],  # Increase number of trees
    'max_depth': [20, 30, 40],         # Increase maximum depth
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Randomized Search CV to find the best hyperparameters
randomized_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist, n_iter=10, cv=3, verbose=2, n_jobs=-1)
randomized_search.fit(X_train_vec, y_train)

# Get the best model
best_rf_model = randomized_search.best_estimator_

# Predictions on the test set
y_test_pred = best_rf_model.predict(X_test_vec)

# Save Predictions to CSV (optional step)
predictions_df = pd.DataFrame({'ID': test_data['ID'], 'GENRE': y_test_pred})
predictions_df.to_csv('test_predictions.csv', index=False)

# Output the results
print("Predictions have been saved to 'test_predictions.csv'")

# Output model performance on training data
y_train_pred = best_rf_model.predict(X_train_vec)
print("\nClassification Report on Training Data:")
print(classification_report(y_train, y_train_pred))

# Allow user to input movie descriptions and predict genres
while True:
    print("\nEnter a movie description to predict its genre (or type 'exit' to quit):")
    user_input = input("Movie description: ")
    
    if user_input.lower() == 'exit':
        break
    
    # Preprocess the user's input, vectorize it, and predict the genre
    user_input_processed = preprocess_text(user_input)
    user_input_vec = vectorizer.transform([user_input_processed])  # Transform the input text
    predicted_genre = best_rf_model.predict(user_input_vec)  # Predict the genre
    
    # Output the predicted genre
    print(f"Predicted Genre: {predicted_genre[0]}")