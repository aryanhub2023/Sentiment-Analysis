# Sentiment Analysis Project

This project is focused on training a sentiment analysis model to classify text into three categories: Negative, Positive, and Neutral. The dataset used consists of labeled text reviews, and the model is trained using a Naive Bayes classifier.

## Project Structure

Sentiment-Analysis/
│
├── balance_data.py        # Script to balance the training and testing datasets
├── script.py              # Main script for training and evaluating sentiment analysis model
├── sentiment_inference.py # Script for predicting sentiment on new reviews
├── shuffle_lines.py       # Script for shuffling the lines in train.txt and test.txt files
├── dataset/
│   ├── train.txt          # Training dataset (shuffled after balancing)
│   ├── test.txt           # Test dataset (shuffled after balancing)
├── X_train_tfidf.pkl      # Saved TF-IDF vectorized training data (if generated)
├── X_test_tfidf.pkl       # Saved TF-IDF vectorized test data (if generated)
├── tfidf_vectorizer.pkl   # Saved TF-IDF vectorizer model
├── sentiment_model.pkl    # Saved Naive Bayes model after training
└── README.md              # This file

## Steps to Run the Project

### 1. Prepare the Dataset

- You need two text files: `train.txt` for training data and `test.txt` for testing data.
- Each line in these files should be in the format:
  ```
  <label> <review>
  ```
  where:
  - `<label>` is `0` for Negative, `1` for Positive, and `2` for Neutral.
  - `<review>` is the text review.

Example of `train.txt` or `test.txt` lines:
```
0 The product quality is terrible.
1 I love this phone! Works great.
2 The product is average and works as expected.
```

### 2. Balance the Dataset

If your dataset is unbalanced (e.g., there are fewer Neutral reviews than Positive or Negative), you can balance the dataset by oversampling the Neutral reviews using `balance_data.py`.

Run the balancing script:

```bash
python balance_data.py
```

This will oversample Neutral reviews and save the balanced dataset back into the same `train.txt` and `test.txt` files.

### 3. Shuffle the Dataset

After balancing, it's important to shuffle the lines in your dataset to ensure that the labels are randomly distributed.

Run the `shuffle_lines.py` script to shuffle the `train.txt` and `test.txt` files:

```bash
python shuffle_lines.py
```

This will shuffle the lines in both `train.txt` and `test.txt` files and overwrite them.

### 4. Train and Evaluate the Model

Once the dataset is prepared, you can train the sentiment analysis model using the `script.py` file.

To train the model, run:

```bash
python script.py
```

This script will:
- Load and clean the dataset.
- Vectorize the text data using TF-IDF.
- Train a Naive Bayes classifier.
- Evaluate the model on the test dataset.
- Save the trained model and vectorizer for future use.

### 5. Predict Sentiment for New Reviews

You can use the `sentiment_inference.py` script to predict the sentiment of new text reviews using the trained model.

To predict sentiment for new reviews:

1. Edit the `new_reviews` list in `sentiment_inference.py` with the reviews you want to analyze.
2. Run the script:

```bash
python sentiment_inference.py
```

This will output the predicted sentiment for each review.

### 6. Requirements

Ensure you have the necessary libraries installed. You can install them using:

```bash
pip install -r requirements.txt
```

Here’s a list of key libraries:
- `pandas`
- `scikit-learn`
- `joblib`

### 7. Notes

- The dataset is expected to be in a simple text format with space-separated labels and reviews.
- The TF-IDF vectorizer and Naive Bayes model are saved after training to avoid retraining every time you need to make predictions.
- The `balance_data.py` script ensures that the dataset is balanced, especially if you have fewer Neutral reviews.

## Conclusion

This project demonstrates a simple sentiment analysis pipeline using machine learning. It covers data preparation, balancing, shuffling, training, evaluation, and prediction. The model can be further tuned or extended to include other machine learning models or text preprocessing techniques.