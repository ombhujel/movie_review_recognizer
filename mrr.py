import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump, load
import os

# Reads reviews from all text files within a directory, preprocesses them, and returns a list.
def preprocess_data(data_dir):
  reviews = []
  for filename in os.listdir(data_dir):
    filepath = os.path.join(data_dir, filename)
    with open(filepath, 'r') as f:
      review = f.read().strip().lower()
      reviews.append(review)
  return reviews

# Load data (replace paths with yours)
lpos_train = preprocess_data("/train/lpos")
lneg_train = preprocess_data("/train/lneg")
lpos_test = preprocess_data("/test/lpos")
lneg_test = preprocess_data("/test/lneg")

# Combine reviews and labels
train_data = lpos_train + lneg_train
train_labels = [1] * len(lpos_train) + [0] * len(lneg_train)
test_data = lpos_test + lneg_test
test_labels = [1] * len(lpos_test) + [0] * len(lneg_test)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=2000)
train_features = vectorizer.fit_transform(train_data)
test_features = vectorizer.transform(test_data)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(train_features, train_labels)

# Make predictions on test data
predictions = model.predict(test_features)

# Evaluate accuracy
accuracy = accuracy_score(test_labels, predictions)
print("Test Accuracy:", accuracy)

# Save the model for future use
dump(model, "sentiment_model.pkl")
dump(vectorizer, "vectorizer.pkl")  
