# Spam Email Classifier

## Project Overview
This project is a Machine Learning application that classifies email messages as **Spam** or **Not Spam (Ham)** using Natural Language Processing (NLP).

The application uses TF-IDF for text feature extraction and a Multinomial Naive Bayes classifier for prediction. A user-friendly web interface is built using Streamlit.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Streamlit
- Joblib

## Dataset
The project uses the Enron Spam Dataset.

Dataset source: https://github.com/MWiechmann/enron_spam_data

- Total email records: 33,716
- Spam emails: 17,171
- Not Spam emails: 16,545

## Machine Learning Workflow
1. Load and explore the dataset.
2. Handle missing values.
3. Combine email subject and message.
4. Clean and preprocess email text.
5. Split the dataset into training and testing sets.
6. Convert text into numerical features using TF-IDF.
7. Train a Multinomial Naive Bayes classifier.
8. Evaluate the model.
9. Save the trained model and TF-IDF vectorizer.
10. Build a Streamlit web application.

## Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 98.47% |
| Precision | 98.01% |
| Recall | 99.01% |
| F1-Score | 98.51% |

## Project Structure

```text
Spam-Email-Classifier/
├── notebooks/
│   └── spam_email_classification.ipynb
├── models/
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
├── dataset/
├── screenshots/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder

```bash
cd Spam-Email-Classifier
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment (Windows)

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python -m streamlit run app.py
```

## How to Use
1. Open the Streamlit application in your browser.
2. Enter or paste an email message.
3. Click the **Predict Email** button.
4. The application displays whether the email is Spam or Not Spam.

## Screenshots

Application screenshots are available in the `screenshots` folder.

## Author
Machine Learning Project — Spam Email Classification
