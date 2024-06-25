from flask import Flask, render_template, request
from joblib import load

# Load the saved model and vectorizer
model = load("sentiment_model.pkl")
vectorizer = load("vectorizer.pkl")

app = Flask(__name__)
app.template_folder = "templates"  

@app.route("/", methods=["GET", "POST"])
def analyze_sentiment():
  if request.method == "POST":
    review_text = request.form["review"]
    # Preprocess the review (optional, based on your preprocessing steps)
    review_text = review_text.strip().lower()
    # Transform the review using the fitted vectorizer
    review_features = vectorizer.transform([review_text])
    # Make a prediction using the loaded model
    prediction = model.predict(review_features)[0]
    sentiment = "Positive Review" if prediction == 1 else "Negative Review"
    return render_template("index.html", sentiment=sentiment, review=review_text)
  return render_template("index.html")

if __name__ == "__main__":
  app.run(debug=True)

