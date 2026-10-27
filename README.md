# Titanic Survival Predictor

A simple Machine Learning web application that predicts whether a Titanic passenger was likely to survive based on passenger details such as age, sex, passenger class, ticket fare, and family information.

The project uses a **Random Forest Classifier** trained on the Titanic dataset and a Flask backend to connect the trained model to a user-friendly HTML and CSS interface.

## Features

- Predict passenger survival using a trained Machine Learning model.
- Input passenger details through a simple web form.
- Display the predicted outcome and predicted-class probability.
- Handle missing values and encode categorical data during preprocessing.
- Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
- Save and reuse the trained model without retraining every time.
- Includes a Google Colab notebook documenting the model training workflow.

## Tech Stack

- **Frontend:** HTML, CSS
- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn, Random Forest Classifier
- **Data Processing:** Pandas, NumPy
- **Model Serialization:** Joblib
- **Dataset:** Titanic passenger dataset

## Project Structure

```text
titanic-survival-predictor/
│
├── app.py
├── titanic_model.pkl
├── titanic_features.pkl
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── Titanic_Survival_Prediction.ipynb
```

The notebook filename may differ depending on the name used in your repository.

## Machine Learning Workflow

1. **Dataset loading:** Load the Titanic dataset using Pandas.
2. **Data exploration:** Examine the dataset, missing values, and survival distribution.
3. **Feature selection:** Select passenger class, sex, age, family information, fare, and embarkation port.
4. **Data preprocessing:** Fill missing values and convert categorical features into numerical columns using one-hot encoding.
5. **Train-test split:** Divide the dataset into training and testing sets.
6. **Model training:** Train a Random Forest Classifier using the training data.
7. **Model evaluation:** Evaluate predictions on unseen test data.
8. **Model saving:** Save the trained model and feature-column order as `.pkl` files.
9. **Web integration:** Load the saved model through Flask and use it to generate predictions from form inputs.

## Model Performance

The model achieved the following results on the test set:

| Metric | Result |
|---|---:|
| Accuracy | 81.56% |
| Precision — Did Not Survive | 82% |
| Recall — Did Not Survive | 89% |
| F1-score — Did Not Survive | 0.86 |
| Precision — Survived | 80% |
| Recall — Survived | 70% |
| F1-score — Survived | 0.74 |

These results are from the notebook's particular train-test split. They do not guarantee the same performance on other datasets.

## Installation and Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd titanic-survival-predictor
```

Replace `<your-repository-url>` with your actual GitHub repository URL.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Ensure the model files exist

Make sure these files are present in the project root:

- `titanic_model.pkl`
- `titanic_features.pkl`

You can generate them by running the Google Colab notebook or download the saved files produced during training.

### 4. Run the application

```bash
python app.py
```

### 5. Open the website

Visit:

```text
http://127.0.0.1:5000
```

Enter the passenger details and click **Predict Survival** to view the result.

## Google Colab Notebook

The included `.ipynb` notebook contains the code for:

- Loading and exploring the Titanic dataset.
- Preprocessing passenger data.
- Training the Random Forest model.
- Evaluating model performance.
- Inspecting feature importance.
- Testing predictions on a sample passenger.
- Exporting the trained model files.

Open the notebook in [Google Colab](https://colab.research.google.com/) to inspect or rerun the training workflow.

## Limitations

- The model uses a relatively small historical dataset.
- Prediction quality depends on the selected features and training data.
- The predicted probability is an estimate from the trained model, not a guarantee of survival.
- Model performance may vary with different data splits and configurations.

## Future Improvements

- Improve feature engineering and model evaluation.
- Add interactive data visualizations.
- Compare Random Forest with Logistic Regression and other classifiers.
- Deploy the application online.
- Improve input validation and automated testing.

## Disclaimer

This project is intended for educational purposes and demonstrates a basic end-to-end Machine Learning workflow. It is not intended for real-world survival predictions.

---

**Built with Python, Flask, HTML, CSS, and Scikit-learn.**
