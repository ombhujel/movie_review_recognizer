import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump, load
import os

def preprocess_data(data_dir):
  """
  Reads reviews from all text files within a directory, preprocesses them, and returns a list.

  Args:
      data_dir: Path to the directory containing review files.

  Returns:
      A list of preprocessed reviews.
  """
  reviews = []
  for filename in os.listdir(data_dir):
    filepath = os.path.join(data_dir, filename)
    with open(filepath, 'r') as f:
      review = f.read().strip().lower()
      reviews.append(review)
  return reviews

# Load data (replace paths with yours)
lpos_train = preprocess_data("/home/om_linux/mrr/train/lpos")
lneg_train = preprocess_data("/home/om_linux/mrr/train/lneg")
lpos_test = preprocess_data("/home/om_linux/mrr/test/lpos")
lneg_test = preprocess_data("/home/om_linux/mrr/test/lneg")

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

# Save the model for future use (optional)
dump(model, "sentiment_model.pkl")
dump(vectorizer, "vectorizer.pkl")  # Added line to save the vectorizer

# Example usage of saved model (optional)
def predict_sentiment(new_review, model_path="sentiment_model.pkl"):
  """
  Predicts sentiment of a new review using the saved model.

  Args:
      new_review: The review text to classify.
      model_path: Path to the saved model (default: sentiment_model.pkl).

  Returns:
      "Positive Review" or "Negative Review" based on the prediction.
  """
  loaded_model = load(model_path)
  new_features = vectorizer.transform([new_review])
  prediction = loaded_model.predict(new_features)
  if prediction[0] == 1:
    return "Positive Review"
  else:
    return "Negative Review"

# Example usage
new_review = "This product is excellent!"
sentiment = predict_sentiment(new_review)
print(sentiment)

