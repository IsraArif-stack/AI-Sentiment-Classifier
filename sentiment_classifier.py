import pandas as pd
import nltk
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Load dataset
data = pd.read_csv("data/sentiment_data.csv")

print("Dataset loaded successfully!")
print(data.head())

# Initialize NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenize words
    words = text.split()

    # Remove stopwords and apply lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    # Join words back together
    return " ".join(words)


# Apply preprocessing
data["clean_text"] = data["text"].apply(preprocess_text)

print("\nOriginal vs Cleaned Text:")
print(data[["text", "clean_text", "sentiment"]].head(10))
from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Convert cleaned text into numerical features
X = vectorizer.fit_transform(data["clean_text"])

# Sentiment labels
y = data["sentiment"]

print("\nTF-IDF conversion successful!")
print("Number of text samples:", X.shape[0])
print("Number of features:", X.shape[1])
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nData split completed!")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Create the classification model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["negative", "neutral", "positive"],
    yticklabels=["negative", "neutral", "positive"]
)

plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Sentiment")
plt.title("Sentiment Classification Confusion Matrix")

plt.tight_layout()

# Save visualization
plt.savefig("outputs/confusion_matrix.png")

plt.show()

print("\nConfusion matrix saved to outputs/confusion_matrix.png")
# Test the model with custom sentences

print("\n--- Custom Sentiment Prediction ---")

while True:
    user_text = input("\nEnter a sentence (or type 'exit' to stop): ")

    if user_text.lower() == "exit":
        print("Prediction system closed.")
        break

    # Preprocess the user input
    cleaned_text = preprocess_text(user_text)

    # Convert text using the trained TF-IDF vectorizer
    text_vector = vectorizer.transform([cleaned_text])

    # Predict sentiment
    prediction = model.predict(text_vector)[0]

    print("Predicted Sentiment:", prediction.upper())
    # Save trained model and TF-IDF vectorizer

joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\nModel saved successfully!")
print("Saved: models/sentiment_model.pkl")
print("Saved: models/tfidf_vectorizer.pkl")