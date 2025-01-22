---
# Genre Classification with Random Forest Classifier

This repository contains a machine learning model to classify movie descriptions into genres using a Random Forest Classifier. The implementation uses TF-IDF vectorization and hyperparameter tuning for better accuracy and performance.

## Features
- Preprocessing of text data (punctuation removal, stopword removal, lemmatization).
- TF-IDF vectorization with trigrams.
- Random Forest Classifier with hyperparameter tuning using `RandomizedSearchCV`.
- Handles class imbalance with `class_weight='balanced'`.
- Allows user interaction for real-time predictions.

---

## Installation and Setup

### Prerequisites
Ensure you have the following installed:
- Python 3.7+
- NLTK
- scikit-learn
- pandas
- scipy

### Steps
1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_folder>
    ```

2. Install required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the NLTK resources used:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    ```

4. Place the dataset files (`train_data.txt` and `test_data.txt`) in the `data/` directory or update the paths in the script.

---

## Dataset
### Training Data
- `train_data.txt`: Contains columns `ID`, `TITLE`, `GENRE`, and `DESCRIPTION` separated by `:::`.

### Test Data
- `test_data.txt`: Contains columns `ID`, `TITLE`, and `DESCRIPTION` separated by `:::` (no `GENRE` column).

---

## Usage
1. Run the script to train the model, predict genres, and save results to a CSV:
    ```bash
    python genre_classifier.py
    ```

2. To predict genres interactively, follow the on-screen instructions to input a movie description.

---

## File Descriptions
- `genre_classifier.py`: The main script for training, testing, and user interaction.
- `test_predictions.csv`: The output file containing predicted genres for the test dataset.
- `requirements.txt`: List of Python dependencies.

---

## Key Functions
- **`preprocess_text(text)`**: Cleans and preprocesses the input text (removes punctuation, stopwords, and lemmatizes words).
- **TF-IDF Vectorization**: Converts text descriptions into numerical representations for model training and predictions.
- **Random Forest Classifier**: A robust machine learning model used for classification.

---

## Example Input and Output

### Example Movie Description
```
A thrilling space adventure where a group of explorers travel beyond our galaxy to uncover the secrets of the universe.
```

### Predicted Genre
```
Science Fiction
```

---

## Hyperparameter Tuning
- **Parameters Tuned**:
  - `n_estimators`: Number of trees in the forest.
  - `max_depth`: Maximum depth of the trees.
  - `min_samples_split`: Minimum samples required to split an internal node.
  - `min_samples_leaf`: Minimum samples required to be at a leaf node.

- **Method**: RandomizedSearchCV with 3-fold cross-validation and 10 iterations.

---

## Results
- **Training Data**: Achieved high accuracy with well-balanced class distribution.
- **Test Data**: Predictions saved in `test_predictions.csv`.

---

## Contribution
Contributions are welcome! Feel free to create a pull request or open an issue to report bugs or suggest features.

---

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

---

## Acknowledgments
- NLTK for text preprocessing.
- scikit-learn for machine learning tools.

---
