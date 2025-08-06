import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


# Download stopwords (only once)
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv("TASK4/data/spam.csv", encoding='latin1')

# Remove unwanted columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']  # Rename for clarity

# Show updated dataset info
print("✅ Cleaned Data:")
print(df.head())

# Check for missing values
print("\n🔍 Missing values:\n", df.isnull().sum())


# Define preprocessing function
def preprocess(text):
    # 1. Lowercase the text
    text = text.lower()
    
    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 3. Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    
    # 4. Join cleaned words back
    return " ".join(words)

# Apply preprocessing to each message
df['cleaned_message'] = df['message'].apply(preprocess)

# Show results
print("✅ Sample cleaned messages:")
print(df[['message', 'cleaned_message']].head())

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Prepare labels (ham = 0, spam = 1)
y = df['label'].map({'ham': 0, 'spam': 1})

# Convert cleaned messages to TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_message'])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Show performance
print("\n📊 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# Manual Testing: Predict your own message
print("\n🔍 Manual Testing - Predict your own message:")

# 1. Input your message here
custom_message = "Congratulations! You've won a free ticket. Click here to claim."

# 2. Preprocess the message just like the dataset
custom_cleaned = preprocess(custom_message)

# 3. Convert to TF-IDF vector using the same vectorizer
custom_vector = vectorizer.transform([custom_cleaned])

# 4. Predict using your trained model
prediction = model.predict(custom_vector)

# 5. Show result
print("🧠 Prediction:", "Spam" if prediction[0] == 1 else "Ham")
