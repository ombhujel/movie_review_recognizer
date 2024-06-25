from joblib import load  # For loading model
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model and vectorizer (assuming they are saved as sentiment_model.pkl and vectorizer.pkl)
loaded_model = load("sentiment_model.pkl")
vectorizer = load("vectorizer.pkl")  # Load the fitted vectorizer

def predict_sentiment(new_review):
  """
  Predicts sentiment of a new review using the saved model.

  Args:
      new_review: The review text to classify.

  Returns:
      "Positive Review" or "Negative Review" based on the prediction.
  """
  new_features = vectorizer.transform([new_review])
  prediction = loaded_model.predict(new_features)
  if prediction[0] == 1:
    return "Positive Review"
  else:
    return "Negative Review"

# Example usage
new_review = "This movie was a disappointment."
sentiment = predict_sentiment(new_review)
print(sentiment)

