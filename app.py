
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

FEATURES = ["pclass", "sex", "age", "fare"]
MODEL_PATH = Path(__file__).parent / "model.joblib"


# Load model once and cache it
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found.")

    return joblib.load(MODEL_PATH)


# App heading
st.title("🚢 Titanic Survival Predictor")

st.write(
    "Enter passenger details to estimate the probability "
    "of survival using a Decision Tree machine learning model."
)

st.divider()

# Passenger inputs
st.subheader("Passenger Details")

pclass = st.selectbox(
    "Passenger Ticket Class",
    options=[1, 2, 3],
    index=2,
    help="1 = First Class, 2 = Second Class, 3 = Third Class"
)

sex = st.radio(
    "Sex",
    options=["Female", "Male"],
    horizontal=True
)

age = st.slider(
    "Passenger Age",
    min_value=1,
    max_value=80,
    value=30
)

fare = st.slider(
    "Ticket Fare",
    min_value=0,
    max_value=500,
    value=30
)

st.divider()

# Prediction
if st.button("Predict Survival", type="primary", use_container_width=True):

    try:
        model = load_model()

        # Convert input into the same format used during training
        sex_encoded = 1 if sex == "Female" else 0

        input_data = pd.DataFrame(
            [[pclass, sex_encoded, age, fare]],
            columns=FEATURES
        )

        # Generate prediction and probability
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        # Find the probability belonging to class 1
        survival_index = list(model.classes_).index(1)
        survival_probability = probabilities[survival_index]

        st.subheader("Prediction Result")

        st.metric(
            label="Estimated Survival Probability",
            value=f"{survival_probability:.1%}"
        )

        st.progress(float(survival_probability))

        if prediction == 1:
            st.success("The model predicts: Survived")
        else:
            st.error("The model predicts: Did not survive")

        st.caption(
            "This is a machine learning prediction based on historical "
            "Titanic passenger data. It is an educational example, "
            "not a factual prediction or advice."
        )

    except FileNotFoundError:
        st.error(
            "The model file is missing. Please upload model.joblib "
            "to the project repository."
        )

    except Exception:
        st.error(
            "Something went wrong while generating the prediction. "
            "Please check the model file and try again."
        )

st.divider()

st.caption(
    "Model: Decision Tree Classifier | "
    "Dataset: Seaborn Titanic Dataset | "
    "Features: Passenger Class, Sex, Age and Fare"
)
