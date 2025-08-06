import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# STEP 1: Load raw data
df_raw = pd.read_csv("TASK1/data/Genre Classification Dataset/train_data.txt", sep="\t", header=None)
print("✅ Loaded using tab (\\t) separator:")
print("➡ First row:\n", df_raw.iloc[0])
print("➡ Shape:", df_raw.shape)

# STEP 2: Split column by delimiter " ::: "
# Each row has 1 string like "1 ::: title ::: genre ::: plot"
df_split = df_raw[0].str.split(" ::: ", expand=True)
df_split.columns = ['id', 'title', 'genre', 'plot']

# STEP 3: Filter useful columns
df = df_split[['genre', 'plot']].dropna()
print("\n✅ Sample after splitting and filtering:")
print(df.head())

# STEP 4: Convert plot to numeric vectors using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['plot'])
y = df['genre']

# STEP 5: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 6: Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# STEP 7: Evaluate
y_pred = model.predict(X_test)
print("\n📊 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))


