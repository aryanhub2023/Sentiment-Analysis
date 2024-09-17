# import pandas as pd
# import re
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, classification_report
# import joblib  # For saving and loading models and preprocessed data
# import os  # To check if files exist

# # Function to load dataset from a file                                
# def load_dataset(file_path): 
#     print(f"Loading dataset from {file_path}...")
#     reviews = []
#     labels = []
    
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             label, review = line.split(' ', 1)  # Split the label and review
#             label = int(label)  # Convert label to integer (0 or 1)
#             reviews.append(review.strip())  # Remove any leading/trailing whitespace
#             labels.append(label)
    
#     print(f"Finished loading dataset from {file_path}. Total reviews loaded: {len(reviews)}")
#     return pd.DataFrame({'review': reviews, 'sentiment': labels})

# # Load the train and test datasets
# # Give the actual path of your datasets
# print("Loading train and test datasets...")
# train_file_path = 'dataset\train.txt'
# test_file_path = 'dataset\test.txt'

# train_data = load_dataset(train_file_path)
# test_data = load_dataset(test_file_path)
# print("Datasets loaded successfully!")

# # Clean the text data (remove special characters, convert to lowercase)
# print("Cleaning the text data...")

# def clean_text(text):
#     text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
#     text = text.lower()  # Convert to lowercase
#     text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
#     return text

# train_data['cleaned_review'] = train_data['review'].apply(clean_text)
# test_data['cleaned_review'] = test_data['review'].apply(clean_text)
# print("Text cleaning completed!")

# # Check if preprocessed data exists, else vectorize the text data using TF-IDF
# preprocessed_data_exists = os.path.exists('X_train_tfidf.pkl') and os.path.exists('X_test_tfidf.pkl') and os.path.exists('tfidf_vectorizer.pkl')

# if preprocessed_data_exists:
#     print("Loading preprocessed TF-IDF data...")
#     X_train = joblib.load('X_train_tfidf.pkl')
#     X_test = joblib.load('X_test_tfidf.pkl')
#     vectorizer = joblib.load('tfidf_vectorizer.pkl')
#     print("Preprocessed TF-IDF data loaded!")
# else:
#     print("Vectorizing the text data using TF-IDF...")
#     vectorizer = TfidfVectorizer(max_features=10000)  # Limit to 10,000 features
#     X_train = vectorizer.fit_transform(train_data['cleaned_review'])  # Transform train data
#     X_test = vectorizer.transform(test_data['cleaned_review'])  # Transform test data
#     print("Text data vectorization completed!")

#     # Save preprocessed data
#     print("Saving preprocessed data...")
#     joblib.dump(X_train, 'X_train_tfidf.pkl')
#     joblib.dump(X_test, 'X_test_tfidf.pkl')
#     joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
#     print("Preprocessed data saved!")

# y_train = train_data['sentiment']
# y_test = test_data['sentiment']

# # Check if the trained model exists, else train a Naive Bayes model
# model_exists = os.path.exists('sentiment_model.pkl')

# if model_exists:
#     print("Loading trained Naive Bayes model...")
#     model = joblib.load('sentiment_model.pkl')
#     print("Trained model loaded!")
# else:
#     print("Training the Naive Bayes model...")
#     model = MultinomialNB()
#     model.fit(X_train, y_train)
#     print("Model training completed!")

#     # Save the trained model
#     print("Saving trained model...")
#     joblib.dump(model, 'sentiment_model.pkl')
#     print("Trained model saved!")

# # Evaluate the model
# print("Evaluating the model on the test dataset...")
# y_pred = model.predict(X_test)

# # Print accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Model Accuracy: {accuracy * 100:.2f}%")

# # Print classification report
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# print("Script execution completed!")
















import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib  # For saving and loading models and preprocessed data
import os  # To check if files exist

# Function to load dataset from a file 
def load_dataset(file_path): 
    print(f"Loading dataset from {file_path}...")
    reviews = []
    labels = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            label, review = line.split(' ', 1)  # Split the label and review
            label = int(label)  # Convert label to integer (0, 1, or 2)
            reviews.append(review.strip())  # Remove any leading/trailing whitespace
            labels.append(label)
    
    print(f"Finished loading dataset from {file_path}. Total reviews loaded: {len(reviews)}")
    return pd.DataFrame({'review': reviews, 'sentiment': labels})

# Load the train and test datasets
print("Loading train and test datasets...")
train_file_path = 'dataset/train.txt'
test_file_path = 'dataset/test.txt'

train_data = load_dataset(train_file_path)
test_data = load_dataset(test_file_path)
print("Datasets loaded successfully!")


# # Check the counts of the data.........
# print("Training data label distribution:")
# print(train_data['sentiment'].value_counts())

# print("Test data label distribution:")
# print(test_data['sentiment'].value_counts())




# Clean the text data (remove special characters, convert to lowercase)
print("Cleaning the text data...")

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text

train_data['cleaned_review'] = train_data['review'].apply(clean_text)
test_data['cleaned_review'] = test_data['review'].apply(clean_text)
print("Text cleaning completed!")

# Check if preprocessed data exists, else vectorize the text data using TF-IDF
preprocessed_data_exists = os.path.exists('X_train_tfidf.pkl') and os.path.exists('X_test_tfidf.pkl') and os.path.exists('tfidf_vectorizer.pkl')

if preprocessed_data_exists:
    print("Loading preprocessed TF-IDF data...")
    X_train = joblib.load('X_train_tfidf.pkl')
    X_test = joblib.load('X_test_tfidf.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("Preprocessed TF-IDF data loaded!")
else:
    print("Vectorizing the text data using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=10000)  # Limit to 10,000 features
    X_train = vectorizer.fit_transform(train_data['cleaned_review'])  # Transform train data
    X_test = vectorizer.transform(test_data['cleaned_review'])  # Transform test data
    print("Text data vectorization completed!")

    # Save preprocessed data
    print("Saving preprocessed data...")
    joblib.dump(X_train, 'X_train_tfidf.pkl')
    joblib.dump(X_test, 'X_test_tfidf.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("Preprocessed data saved!")

y_train = train_data['sentiment']
y_test = test_data['sentiment']





# Check if the trained model exists, else train a Naive Bayes model
model_exists = os.path.exists('sentiment_model.pkl')

if model_exists:
    print("Loading trained Naive Bayes model...")
    model = joblib.load('sentiment_model.pkl')
    print("Trained model loaded!")
else:
    print("Training the Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train, y_train)
    print("Model training completed!")

    # Save the trained model
    print("Saving trained model...")
    joblib.dump(model, 'sentiment_model.pkl')
    print("Trained model saved!")

# Evaluate the model
print("Evaluating the model on the test dataset...")
y_pred = model.predict(X_test)

# Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred,labels=[0, 1, 2], target_names=['Negative', 'Positive', 'Neutral']))

print("Script execution completed!")
