---
# Fraud Detection Using Random Forest Classifier

This repository contains a machine learning project to detect fraudulent transactions using a Random Forest Classifier. The dataset is processed and analyzed with features like scaling, one-hot encoding, and model evaluation metrics.

---

## Features
- Data cleaning and preprocessing for better model training.
- Feature scaling using `StandardScaler`.
- Fraud prediction using Random Forest Classifier.
- Evaluation metrics: accuracy, classification report, confusion matrix, ROC curve, and AUC score.
- Interactive prediction for user-defined transaction details.

---

## Installation and Setup

### Prerequisites
Ensure you have the following installed:
- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

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

3. Place the `fraudTrain.csv` and `fraudTest.csv` files in the appropriate folder or update their paths in the script.

---

## Dataset
### Description
The dataset contains transactional data for fraud detection. It includes both legitimate and fraudulent transactions labeled under the `is_fraud` column.

### Format
- `is_fraud`: Target variable (1 = Fraud, 0 = Legitimate)
- Features include transaction details, demographic information, and location.

### Dropped Columns
The following columns are removed as they are not useful for prediction:
- `Unnamed: 0`, `trans_date_trans_time`, `cc_num`, `merchant`, `first`, `last`, `street`, `city`, `state`, `zip`, `lat`, `long`, `job`, `dob`, `trans_num`, `unix_time`, `merch_lat`, `merch_long`

---

## Usage
1. Run the script to train the model and evaluate its performance:
    ```bash
    python fraud_detection.py
    ```

2. Enter transaction details when prompted to predict fraudulence.

---

## File Descriptions
- `fraud_detection.py`: The main script for data preprocessing, model training, evaluation, and prediction.
- `fraudTrain.csv` and `fraudTest.csv`: Training and testing datasets.
- `requirements.txt`: List of Python dependencies.

---

## Key Functions
- **Data Preprocessing**: Cleans and encodes the data for machine learning.
- **Random Forest Classifier**: Trains a model to detect fraud.
- **Evaluation Metrics**:
  - Confusion Matrix
  - ROC Curve and AUC
  - Classification Report
- **`predict_fraud(transaction_data)`**: Predicts if a transaction is fraudulent based on user input.

---

## Example Input and Output

### Example Input
```plaintext
Enter transaction details for fraud prediction:
Age: 45
Income: 75000
Transaction Amount: 150
```

### Example Output
```plaintext
The transaction is: Legitimate
```

---

## Results
- **Accuracy**: Model performance on the test set.
- **AUC**: High Area Under Curve score for ROC Curve, indicating good performance.

---

## Contribution
Contributions are welcome! Feel free to create a pull request or open an issue to report bugs or suggest features.

---

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

---

## Acknowledgments
- scikit-learn for machine learning tools.
- matplotlib and seaborn for data visualization.

---
