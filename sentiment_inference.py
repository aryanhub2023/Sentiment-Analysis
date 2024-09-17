# # Import necessary libraries
# import re
# import joblib  # For loading models and preprocessed data

# # Load the pre-trained model and TF-IDF vectorizer
# print("Loading pre-trained model and TF-IDF vectorizer...")
# model = joblib.load('sentiment_model.pkl')
# vectorizer = joblib.load('tfidf_vectorizer.pkl')
# print("Model and vectorizer loaded successfully!")

# # Define a function to clean the text (same as used earlier)
# def clean_text(text):
#     text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
#     text = text.lower()  # Convert to lowercase
#     text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
#     return text

# # Function to predict the sentiment of new reviews
# def predict_sentiment(new_reviews):
#     print("Cleaning and vectorizing new reviews...")
#     new_reviews_cleaned = [clean_text(review) for review in new_reviews]
#     new_reviews_tfidf = vectorizer.transform(new_reviews_cleaned)
    
#     print("Making predictions...")
#     predictions = model.predict(new_reviews_tfidf)

#     for review, sentiment in zip(new_reviews, predictions):
#         print(f"Review: {review} --> Sentiment: {'Positive' if sentiment == 1 else 'Negative'}")

# # Use the function to predict sentiment on new texts
# new_reviews = ["The item arrived on time, but it's just okay.", "I don't think it's worth the money.", "Absolutely loved it!"]
# predict_sentiment(new_reviews)

# print("Sentiment prediction completed!")







# Import necessary libraries
import re
import joblib  # For loading models and preprocessed data

# Load the pre-trained model and TF-IDF vectorizer
print("Loading pre-trained model and TF-IDF vectorizer...")
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
print("Model and vectorizer loaded successfully!")

# Define a function to clean the text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text

# Function to predict the sentiment of new reviews
def predict_sentiment(new_reviews):
    print("Cleaning and vectorizing new reviews...")
    new_reviews_cleaned = [clean_text(review) for review in new_reviews]
    new_reviews_tfidf = vectorizer.transform(new_reviews_cleaned)
    
    print("Making predictions...")
    predictions = model.predict(new_reviews_tfidf)

    # Map numeric labels to sentiment names
    sentiment_labels = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}

    for review, sentiment in zip(new_reviews, predictions):
        print(f"Review: {review} --> Sentiment: {sentiment_labels[sentiment]}")

# Use the function to predict sentiment on new texts
new_reviews = ["The software update includes several minor improvements.", "I don't think it's worth the money.", "Absolutely loved it!"]
predict_sentiment(new_reviews)

print("Sentiment prediction completed!")
