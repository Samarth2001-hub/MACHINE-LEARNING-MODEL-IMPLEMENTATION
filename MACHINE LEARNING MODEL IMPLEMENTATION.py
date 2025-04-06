from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

 
emails = [
    "win free money now click here",
    "meeting tomorrow at 9am",
    "urgent cash prize claim now",
    "project update for team",
    "limited time offer buy now",
    "lunch plans today?",
    "exclusive deal just for you",
    "weekly status report"
]

labels = [1, 0, 1, 0, 1, 0, 1, 0]

 
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

 
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42
)

 
model = MultinomialNB()
model.fit(X_train, y_train)

 
y_pred = model.predict(X_test)

 
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

 
new_emails = [
    "win a free vacation now",
    "team meeting at 2pm"
]
new_X = vectorizer.transform(new_emails)
predictions = model.predict(new_X)

 
for email, pred in zip(new_emails, predictions):
    status = "spam" if pred == 1 else "not spam"
    print(f"\nEmail: {email}")
    print(f"Prediction: {status}")

 
probabilities = model.predict_proba(new_X)
for email, prob in zip(new_emails, probabilities):
    print(f"\nEmail: {email}")
    print(f"Spam probability: {prob[1]:.2f}")
