# Import libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Sample data
emails = [
    "Win a free iPhone now",
    "Meeting at 11 am tomorrow",
    "Congratulations you won lottery",
    "Project discussion with team",
    "Claim your prize immediately",
    "Please find the attached report"
]

labels = [1, 0, 1, 0, 1, 0]   # 1 = Spam, 0 = Not Spam

# Convert text to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(emails)

# Train model
model = LinearSVC()
model.fit(X, labels)

# Take user input
message = input("Enter Email Message: ")

# Predict
msg_vector = vectorizer.transform([message])
prediction = model.predict(msg_vector)[0]

if prediction == 1:
    print("Result: Spam Email 🚫")
else:
    print("Result: Not Spam Email ✅")
