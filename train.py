import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Step 1: Load your dataset
df = pd.read_csv("training_modified.csv")

# Step 2: Keep only sentiment + tweet columns
df = df[['Sentiment', 'Tweet']]

# Step 3: Convert sentiment 4 → 1 (positive), 0 stays 0 (negative)
df['Sentiment'] = df['Sentiment'].apply(lambda x: 1 if x == 4 else 0)

# Step 4: Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['Tweet'], df['Sentiment'], test_size=0.2, random_state=42)

# Step 5: Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Train logistic regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Step 7: Evaluate model
y_pred = model.predict(X_test_vec)
print("\n📊 Model Evaluation:\n")
print(classification_report(y_test, y_pred))

# Step 8: Save model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\n✅ Model and vectorizer saved successfully!")