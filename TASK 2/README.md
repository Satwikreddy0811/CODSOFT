---
# SMS Spam Classifier using Linear SVM

This repository contains a machine learning model for classifying SMS messages as spam or ham (not spam) using a Linear Support Vector Machine (SVM). The implementation uses TF-IDF vectorization for feature extraction from text.

---

## Features
- Preprocesses SMS text data.
- Uses TF-IDF vectorization to extract features from messages.
- Trains a Linear Support Vector Machine (SVM) classifier.
- Provides a user-friendly interface to classify new SMS messages.

---

## Installation and Setup

### Prerequisites
Ensure you have the following installed:
- Python 3.7+
- scikit-learn
- pandas

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

3. Place the `spam.csv` file in the root folder or update the path in the script.

---

## Dataset
### Description
The dataset (`spam.csv`) contains SMS messages labeled as:
- `ham`: Legitimate messages.
- `spam`: Unwanted or promotional messages.

### Format
The dataset includes two columns:
- `v1`: Label (`ham` or `spam`).
- `v2`: Message content.

---

## Usage
1. Train the SVM classifier and classify SMS messages by running:
    ```bash
    python sms_spam_classifier.py
    ```

2. Input a custom SMS message when prompted to classify it as spam or ham.

---

## File Descriptions
- `sms_spam_classifier.py`: The main script for training the model and classifying messages.
- `spam.csv`: The input dataset containing SMS messages.
- `requirements.txt`: List of Python dependencies.

---

## Key Functions
- **TF-IDF Vectorization**: Converts SMS text into numerical feature vectors.
- **Linear SVM Classifier**: A machine learning model for binary classification.
- **`classify_message(msg)`**: Classifies a single SMS message as spam or ham.

---

## Example Input and Output

### Example Input
```
Enter an SMS message to classify (Spam/Ham): You won a free ticket! Claim it now.
```

### Example Output
```
Result: The message is classified as 'Spam'.
```

---

## Results
- **Training Accuracy**: Achieved using the Linear SVM classifier.
- **Test Accuracy**: Displayed in the classification report.

---

## Contribution
Contributions are welcome! Feel free to create a pull request or open an issue to report bugs or suggest features.

---

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

---

## Acknowledgments
- scikit-learn for machine learning tools.

---
