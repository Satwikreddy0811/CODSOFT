import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset and clean up unnecessary columns
df = pd.read_csv('/Users/satwikreddy/Downloads/spam.csv', encoding='latin-1')
df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})

# Encode the labels: 0 for 'ham', 1 for 'spam'
df['label'] = df['label'].apply(lambda x: 1 if x == 'spam' else 0)

# Split the data into training and testing subsets
X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Transform text data into TF-IDF feature vectors
tfidf = TfidfVectorizer(stop_words='english', max_df=0.9)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train a Support Vector Machine (SVM) classifier
svm_model = LinearSVC()
svm_model.fit(X_train_tfidf, y_train)

# Test the model and display results
y_pred = svm_model.predict(X_test_tfidf)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nDetailed Classification Report:\n")
print(classification_report(y_test, y_pred))

# Define a function for classifying new messages
def classify_message(msg):
    msg_vector = tfidf.transform([msg])
    prediction = svm_model.predict(msg_vector)
    return "Spam" if prediction[0] == 1 else "Ham"

# Take user input and predict its classification
if __name__ == "__main__":
    user_input = input("Enter an SMS message to classify (Spam/Ham): ")
    classification = classify_message(user_input)
    print(f"Result: The message is classified as '{classification}'")
