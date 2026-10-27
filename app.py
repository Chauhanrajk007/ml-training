
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the already-trained model
model = joblib.load("titanic_model.pkl")
feature_columns = joblib.load("titanic_features.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            pclass = int(request.form["pclass"])
            sex = request.form["sex"]
            age = float(request.form["age"])
            sibsp = int(request.form["sibsp"])
            parch = int(request.form["parch"])
            fare = float(request.form["fare"])
            embarked = request.form["embarked"]

            # Validate the input
            if pclass not in [1, 2, 3]:
                raise ValueError("Select a valid passenger class.")

            if sex not in ["male", "female"]:
                raise ValueError("Select a valid sex.")

            if not 0 <= age <= 100:
                raise ValueError("Age must be between 0 and 100.")

            if sibsp < 0 or parch < 0 or fare < 0:
                raise ValueError("Family counts and fare cannot be negative.")

            if embarked not in ["C", "Q", "S"]:
                raise ValueError("Select a valid boarding port.")

            # Create input using the same features used during training
            passenger = {
                "Pclass": pclass,
                "Age": age,
                "SibSp": sibsp,
                "Parch": parch,
                "Fare": fare,
                "Sex_female": int(sex == "female"),
                "Sex_male": int(sex == "male"),
                "Embarked_C": int(embarked == "C"),
                "Embarked_Q": int(embarked == "Q"),
                "Embarked_S": int(embarked == "S"),
            }

            input_df = pd.DataFrame([passenger])
            input_df = input_df.reindex(
                columns=feature_columns, fill_value=0
            )

            # Predict using the saved model
            prediction = int(model.predict(input_df)[0])
            probabilities = model.predict_proba(input_df)[0]

            result = (
                "Likely to Survive"
                if prediction == 1
                else "Unlikely to Survive"
            )

            probability = round(
                float(probabilities[list(model.classes_).index(prediction)])
                * 100,
                2
            )

        except (ValueError, KeyError) as e:
            error = str(e) or "Please enter valid passenger details."

    return render_template(
        "index.html",
        result=result,
        probability=probability,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
