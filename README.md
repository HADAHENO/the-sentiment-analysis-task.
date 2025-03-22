# **Twitter Sentiment Analysis** 🐦🔍  

## **📌 Overview**  
This project focuses on **Twitter Sentiment Analysis** using **Machine Learning models** to classify tweets into sentiment categories. It involves data preprocessing, feature extraction, and training different models, including **Naïve Bayes** and **Random Forest**, to analyze sentiments effectively.  

🔗 **Full Code on Kaggle:** [Twitter Sentiment Analysis](https://www.kaggle.com/code/hudamaher/twiter-sentiment-analysis)  

---

## **📌 Steps Followed**  

### **1️⃣ Import Needed Modules**  
```python
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import nltk  
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize  
from nltk.stem import PorterStemmer  
from sklearn.model_selection import train_test_split  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.naive_bayes import MultinomialNB  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  
import warnings  
warnings.filterwarnings("ignore")
```
🔹 Importing essential libraries for **data handling, visualization, text preprocessing, and machine learning models**.  

---

### **2️⃣ Exploratory Data Analysis (EDA)**  
```python
df = pd.read_csv("twitter_data.csv")  # Load dataset  
print(df.head())  # Display first few rows  
print(df.info())  # Check dataset structure  
print(df.isnull().sum())  # Check missing values  

sns.countplot(x=df["sentiment"])  # Visualize sentiment distribution  
plt.title("Sentiment Distribution")  
plt.show()
```
🔹 Checking dataset properties, missing values, and sentiment distribution.  

---

### **3️⃣ Preprocessing**  
#### **Drop NaN Values**  
```python
df.dropna(inplace=True)
```
🔹 Removing missing values to avoid errors during training.  

#### **Text Cleaning**  
```python
import re  

def clean_text(text):  
    text = text.lower()  # Convert to lowercase  
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs  
    text = re.sub(r"\@\w+|\#", "", text)  # Remove mentions and hashtags  
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation  
    return text  

df["cleaned_text"] = df["text"].apply(clean_text)
```
🔹 Cleaning tweets by **removing URLs, mentions, hashtags, and punctuation**.  

#### **Tokenization & Stemming**  
```python
nltk.download("stopwords")  
stop_words = set(stopwords.words("english"))  
stemmer = PorterStemmer()  

def preprocess_text(text):  
    words = word_tokenize(text)  
    words = [stemmer.stem(word) for word in words if word not in stop_words]  
    return " ".join(words)  

df["processed_text"] = df["cleaned_text"].apply(preprocess_text)
```
🔹 Applying **tokenization, stopword removal, and stemming** for better text representation.  

---

### **4️⃣ Apply Preprocessing Function on Dataframe**  
```python
df["final_text"] = df["processed_text"].apply(lambda x: preprocess_text(x))
```
🔹 Ensuring preprocessing function is applied to all tweets.  

---

### **5️⃣ Encoding Target Column**  
```python
sentiment_mapping = {"Irrelevant": 0, "Neutral": 1, "Negative": 2, "Positive": 3}  
df["sentiment"] = df["sentiment"].map(sentiment_mapping)
```
🔹 Converting sentiment labels into **numerical categories**.  

---

### **6️⃣ Split Data into Train and Test**  
```python
X_train, X_test, y_train, y_test = train_test_split(df["final_text"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"])
```
🔹 Splitting data into **80% training - 20% testing** while maintaining sentiment balance.  

---

### **7️⃣ Machine Learning Models**  

#### **TF-IDF Vectorization**  
```python
vectorizer = TfidfVectorizer(max_features=5000)  
X_train_tfidf = vectorizer.fit_transform(X_train)  
X_test_tfidf = vectorizer.transform(X_test)
```
🔹 Converting text into numerical features using **TF-IDF**.  

#### **Naïve Bayes Model**  
```python
nb_model = MultinomialNB()  
nb_model.fit(X_train_tfidf, y_train)  

y_pred_nb = nb_model.predict(X_test_tfidf)  
print("Naïve Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
```
🔹 **Naïve Bayes** is a **probabilistic classifier** suitable for text classification.  

#### **Random Forest Model**  
```python
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  
rf_model.fit(X_train_tfidf, y_train)  

y_pred_rf = rf_model.predict(X_test_tfidf)  
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
```
🔹 **Random Forest** is an **ensemble model** that handles data noise effectively.  

---

### **8️⃣ Test Model Performance**  
```python
print("Naïve Bayes Classification Report:\n", classification_report(y_test, y_pred_nb))  
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))  

sns.heatmap(confusion_matrix(y_test, y_pred_nb), annot=True, fmt="d", cmap="Blues")  
plt.title("Naïve Bayes Confusion Matrix")  
plt.show()
```
🔹 Evaluating models using **accuracy, precision, recall, F1-score, and confusion matrix**.  

---

### **9️⃣ Apply Preprocessing on New Text & Get Prediction**  
```python
def predict_sentiment(text, model):  
    cleaned = clean_text(text)  
    processed = preprocess_text(cleaned)  
    vectorized = vectorizer.transform([processed])  
    prediction = model.predict(vectorized)[0]  
    return prediction  

text = "I love this product!"  
predicted_sentiment = predict_sentiment(text, rf_model)  
print("Predicted Sentiment:", predicted_sentiment)
```
🔹 Function to preprocess and classify **new tweets** dynamically.  

---

## **📌 Conclusion**  
This project demonstrates an **end-to-end Twitter sentiment analysis pipeline**, from **data cleaning to model training and evaluation**. It provides insights into tweet sentiments and can be extended using deep learning models like **LSTMs or Transformers**.  

🔗 **Full Code on Kaggle:** [Twitter Sentiment Analysis](https://www.kaggle.com/code/hudamaher/twiter-sentiment-analysis)  

🚀 **Feel free to contribute and improve this project!** 🚀  

---

