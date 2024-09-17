import pandas as pd
import random

def load_text_file(file_path):
    """Load text file into a DataFrame."""
    print(f"Loading dataset from {file_path}...")
    reviews = []
    labels = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.split(' ', 1)
            if len(parts) > 1:  # Ensure there is both a label and a review
                label = int(parts[0])
                review = parts[1].strip()
                reviews.append(review)
                labels.append(label)
    
    print(f"Finished loading dataset from {file_path}. Total reviews loaded: {len(reviews)}")
    return pd.DataFrame({'label': labels, 'review': reviews})

def save_to_text_file(data, file_path):
    """Save DataFrame to text file."""
    print(f"Saving dataset to {file_path}...")
    with open(file_path, 'w', encoding='utf-8') as file:
        for _, row in data.iterrows():
            file.write(f"{row['label']} {row['review']}\n")
    print(f"Dataset saved to {file_path}!")

def balance_and_save_dataset(train_file_path, test_file_path):
    # Load datasets
    train_data = load_text_file(train_file_path)
    test_data = load_text_file(test_file_path)
    
    # Print original class distributions
    print("Original Training Data Distribution:")
    print(train_data['label'].value_counts())
    print("Original Test Data Distribution:")
    print(test_data['label'].value_counts())
    
    # Separate neutral reviews
    neutral_train = train_data[train_data['label'] == 2]
    neutral_test = test_data[test_data['label'] == 2]
    
    # Create balanced datasets by sampling
    num_positive = train_data[train_data['label'] == 1].shape[0]
    num_negative = train_data[train_data['label'] == 0].shape[0]
    num_neutral = neutral_train.shape[0]
    
    # Balance training data
    if num_neutral < max(num_positive, num_negative):
        additional_neutral_needed = max(num_positive, num_negative) - num_neutral
        sampled_neutral_train = neutral_train.sample(n=additional_neutral_needed, replace=True, random_state=1)
        balanced_train_data = pd.concat([train_data[train_data['label'] != 2], sampled_neutral_train])
    else:
        balanced_train_data = train_data
    
    # Balance test data
    num_positive_test = test_data[test_data['label'] == 1].shape[0]
    num_negative_test = test_data[test_data['label'] == 0].shape[0]
    num_neutral_test = neutral_test.shape[0]
    
    if num_neutral_test < max(num_positive_test, num_negative_test):
        additional_neutral_needed_test = max(num_positive_test, num_negative_test) - num_neutral_test
        sampled_neutral_test = neutral_test.sample(n=additional_neutral_needed_test, replace=True, random_state=1)
        balanced_test_data = pd.concat([test_data[test_data['label'] != 2], sampled_neutral_test])
    else:
        balanced_test_data = test_data
    
    # Print new class distributions
    print("Balanced Training Data Distribution:")
    print(balanced_train_data['label'].value_counts())
    print("Balanced Test Data Distribution:")
    print(balanced_test_data['label'].value_counts())
    
    # Save balanced datasets back to files
    save_to_text_file(balanced_train_data, train_file_path)
    save_to_text_file(balanced_test_data, test_file_path)

# Define file paths
train_file_path = 'dataset/train.txt'
test_file_path = 'dataset/test.txt'

# Run the balancing function
balance_and_save_dataset(train_file_path, test_file_path)
