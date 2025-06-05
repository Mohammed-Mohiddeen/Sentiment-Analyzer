from flask import Flask, render_template, request
import joblib

# Load trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    sentiment = "Positive \U0001F600" if prediction == 1 else "Negative \U0001F641"
    
    return render_template("index.html", text=text, sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
