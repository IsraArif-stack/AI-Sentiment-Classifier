# Intelligent Multi-Class Sentiment Classifier

A Natural Language Processing (NLP) and Machine Learning project that classifies text into three sentiment categories:

- Positive
- Negative
- Neutral

This project uses NLTK for text preprocessing, TF-IDF for feature extraction, and Logistic Regression for multi-class sentiment classification.

---

## Project Overview

This project implements an end-to-end sentiment classification pipeline that:

1. Loads a labeled sentiment dataset.
2. Cleans and preprocesses textual data.
3. Removes stop-words.
4. Applies lemmatization.
5. Converts text into numerical features using TF-IDF.
6. Splits the dataset into training and testing sets.
7. Trains a Logistic Regression classifier.
8. Evaluates the model using accuracy, precision, recall, and F1-score.
9. Generates a confusion matrix.
10. Saves the trained model and TF-IDF vectorizer.
11. Allows users to test custom sentences.

---

## Features

- Multi-class sentiment classification
- Positive, Negative, and Neutral sentiment categories
- Text preprocessing using NLTK
- Lowercase conversion
- Punctuation removal
- Stop-word removal
- Lemmatization
- TF-IDF vectorization
- Logistic Regression classification
- Accuracy evaluation
- Precision, Recall, and F1-score
- Confusion matrix visualization
- Custom text prediction
- Prediction confidence score
- Saved machine learning model
- Saved TF-IDF vectorizer

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Main programming language |
| Pandas | Dataset loading and manipulation |
| NumPy | Numerical processing |
| NLTK | Natural Language Processing |
| Scikit-learn | Machine Learning |
| Matplotlib | Data visualization |
| Seaborn | Confusion Matrix |
| Joblib | Model persistence |
| VS Code | Development environment |

---

## Dataset

The project uses a balanced dataset containing:

- 150 total text samples
- 50 Positive samples
- 50 Negative samples
- 50 Neutral samples

## NLP Preprocessing

The following preprocessing steps are applied to the text data.

## 1. Lowercase Conversion

All text is converted into lowercase.

## Example:

"I LOVE This Product"

becomes:

"i love this product"
## 2. Punctuation Removal

Punctuation marks and special characters are removed from the text.
## Example:

"Excellent product!"

becomes:

"excellent product"
## 3. Stop-word Removal

Common English stop-words are removed using NLTK.

## Example:

"I am very happy with the service"

becomes approximately:

"happy service"
## 4. Lemmatization

Lemmatization is applied using NLTK's WordNetLemmatizer to normalize words.

## TF-IDF Vectorization

After preprocessing, the cleaned text is converted into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency).

The final dataset contained:

## Total Samples: 150
## TF-IDF Features: 249

## TF-IDF allows the machine learning model to process textual information as numerical features.
Machine Learning Model
Logistic Regression

Logistic Regression is used as the classification algorithm.

The model is configured as:

LogisticRegression(max_iter=1000)

The model is trained using the TF-IDF representation of the training data.
## Project Structure
AI-Sentiment-Classifier/
│
├── data/
│   └── sentiment_data.csv
│
├── models/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── outputs/
│   └── confusion_matrix.png
│
├── sentiment_classifier.py
├── requirements.txt
├── README.md
├── .gitignore
└── venv/
## Conclusion

The Intelligent Multi-Class Sentiment Classifier successfully implements an end-to-end Natural Language Processing and Machine Learning pipeline.

The system processes raw text using preprocessing techniques including lowercase conversion, punctuation removal, stop-word removal, and lemmatization. The processed text is converted into numerical TF-IDF features and classified using Logistic Regression.

The model was trained on 120 samples and evaluated on 30 testing samples. It achieved an overall accuracy of 63.33% and a macro F1-score of 0.62.

The trained model, TF-IDF vectorizer, and confusion matrix are saved for future use
## Author

## Isra Arif

Artificial Intelligence Internship at progree

Project: Intelligent Multi-Class Sentiment Classifier
