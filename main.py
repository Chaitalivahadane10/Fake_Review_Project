import pandas as pd
import re
df=pd.read_csv("reviews.csv")

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    return text

df['clean_review'] = df['review'].apply(clean_text)

print(df[['review','clean_review']])


from sklearn.feature_extraction.text import TfidfVectorizer

# Convert text to numerical form
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_review']).toarray()

y = df['label']

print("Data converted successfully")


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))



import matplotlib.pyplot as plt

df['label'].value_counts().plot(kind='bar')
plt.title("Fake vs Real Reviews")
plt.xlabel("Type")
plt.ylabel("Count")
plt.show()









