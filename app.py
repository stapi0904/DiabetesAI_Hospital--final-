import streamlit as st
import pandas as pd
import pickle
import os


# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="DiabetesAI Hospital",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# FILE PATHS
# =========================================================

MODEL_FILE = "diabetes_model.pkl"
DATA_FILE = "diabetes_prediction_dataset.csv"


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    with open(MODEL_FILE, "rb") as file:
        return pickle.load(file)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    return pd.read_csv(DATA_FILE)


# =========================================================
# CHECK FILES
# =========================================================

if not os.path.exists(MODEL_FILE):

    st.error(
        "diabetes_model.pkl was not found."
    )

    st.stop()


if not os.path.exists(DATA_FILE):

    st.error(
        "diabetes_prediction_dataset.csv was not found."
    )

    st.stop()


# =========================================================
# LOAD FILES
# =========================================================

try:

    saved_model = load_model()

    # The pickle file contains a dictionary.
    # The actual machine learning model is stored under "model".
    model = saved_model["model"]

    # Model accuracy is stored under "accuracy".
    model_accuracy = saved_model["accuracy"]

    data = load_data()

except Exception as e:

    st.error(
        "The system could not load the required files."
    )

    st.error(str(e))

    st.stop()

# =========================================================
# DATASET COLUMNS
# =========================================================

required_columns = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "blood_glucose_level",
    "diabetes"
]

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:

    st.error(
        "Your dataset is missing required columns."
    )

    st.write(
        "Missing:",
        missing_columns
    )

    st.stop()


# =========================================================
# MODEL INFORMATION
# =========================================================

if hasattr(model, "feature_names_in_"):

    model_features = list(
        model.feature_names_in_
    )

else:

    model_features = [
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "smoking_history",
        "bmi",
        "blood_glucose_level"
    ]


# =========================================================
# ENCODING
# =========================================================

def encode_value(
    value,
    column
):

    unique_values = (
        data[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    # Find matching value
    for item in unique_values:

        if str(item).lower() == str(value).lower():

            return unique_values.index(item)

    return 0


# =========================================================
# PREDICTION
# =========================================================

def predict_diabetes(
    patient_id,
    gender,
    age,
    hypertension,
    heart_disease,
    smoking_history,
    bmi,
    blood_glucose_level
):

    # Encode categorical values
    gender_encoded = encode_value(
        gender,
        "gender"
    )

    smoking_encoded = encode_value(
        smoking_history,
        "smoking_history"
    )

    # -----------------------------------------------------
    # Create input
    # -----------------------------------------------------

    input_data = pd.DataFrame({

        "gender": [
            gender_encoded
        ],

        "age": [
            age
        ],

        "hypertension": [
            hypertension
        ],

        "heart_disease": [
            heart_disease
        ],

        "smoking_history": [
            smoking_encoded
        ],

        "bmi": [
            bmi
        ],

        "blood_glucose_level": [
            blood_glucose_level
        ]

    })

    # -----------------------------------------------------
    # Match model features
    # -----------------------------------------------------

    for feature in model_features:

        if feature not in input_data.columns:

            input_data[feature] = 0

    input_data = input_data[
        model_features
    ]

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    prediction = model.predict(
        input_data
    )

    predicted_class = prediction[0]

    # -----------------------------------------------------
    # Probability
    # -----------------------------------------------------

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            input_data
        )

        if hasattr(model, "classes_"):

            classes = list(
                model.classes_
            )

            if 1 in classes:

                index = classes.index(1)

            else:

                index = len(classes) - 1

        else:

            index = 1

        probability = (
            probabilities[0][index]
            * 100
        )

    else:

        if predicted_class == 1:

            probability = 100

        else:

            probability = 0

    # -----------------------------------------------------
    # Risk
    # -----------------------------------------------------

    if probability >= 70:

        risk = "High Risk"

    elif probability >= 40:

        risk = "Moderate Risk"

    else:

        risk = "Low Risk"

    return {
        "Patient ID": patient_id,
        "Age": age,
        "Gender": gender,
        "BMI": bmi,
        "Blood Glucose": blood_glucose_level,
        "Risk": risk,
        "Probability": probability
    }


# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:

    st.session_state.history = []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🩺 DiabetesAI")

    st.caption(
        "Clinical Risk Prediction System"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "New Assessment",
            "Assessment History",
            "Risk Analytics",
            "AI Model",
            "About"
        ]
    )

    st.divider()

    st.success(
        "AI Model Ready"
    )

    st.caption(
        "Academic Prototype"
    )


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.title(
        "Good afternoon"
    )

    st.write(
        "Clinical risk prediction overview"
    )

    st.divider()

    st.info(
        "Demo Dataset — This application contains "
        "sample data for academic demonstration."
    )

    history = st.session_state.history

    total = len(history)

    high = sum(
        x["Risk"] == "High Risk"
        for x in history
    )

    moderate = sum(
        x["Risk"] == "Moderate Risk"
        for x in history
    )

    low = sum(
        x["Risk"] == "Low Risk"
        for x in history
    )

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    st.subheader(
        "Assessment Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Assessments",
            total
        )

    with col2:

        st.metric(
            "High Risk",
            high
        )

    with col3:

        st.metric(
            "Moderate Risk",
            moderate
        )

    with col4:

        st.metric(
            "Low Risk",
            low
        )

    st.divider()

    # -----------------------------------------------------
    # AI ASSESSMENT
    # -----------------------------------------------------

    st.subheader(
        "✦ AI-Powered Assessment"
    )

    st.write(
        "Enter patient information and generate "
        "a diabetes risk prediction."
    )

    st.info(
        "Go to 'New Assessment' in the navigation menu "
        "to start a prediction."
    )

    st.divider()

    # -----------------------------------------------------
    # RECENT RECORDS
    # -----------------------------------------------------

    st.subheader(
        "Recent Assessments"
    )

    if len(history) > 0:

        recent = pd.DataFrame(
            history
        ).tail(5).iloc[::-1]

        display_df = recent.copy()

        display_df["Probability"] = (
            display_df["Probability"]
            .round(2)
            .astype(str)
            + "%"
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No assessment records yet."
        )


# =========================================================
# NEW ASSESSMENT
# =========================================================

elif page == "New Assessment":

    st.title(
        "New Assessment"
    )

    st.write(
        "Enter patient information below."
    )

    st.divider()

    # -----------------------------------------------------
    # PATIENT
    # -----------------------------------------------------

    st.subheader(
        "Patient Information"
    )

    patient_id = st.text_input(
        "Patient ID",
        placeholder="Example: PT-011"
    )

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30
        )

    with col2:

        gender_values = (
            data["gender"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        gender = st.selectbox(
            "Gender",
            gender_values
        )

    st.divider()

    # -----------------------------------------------------
    # CLINICAL
    # -----------------------------------------------------

    st.subheader(
        "Clinical Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=70.0,
            value=24.0,
            step=0.1
        )

    with col2:

        blood_glucose = st.number_input(
            "Blood Glucose Level",
            min_value=50,
            max_value=400,
            value=100
        )

    with col3:

        hypertension = st.selectbox(
            "Hypertension",
            ["No", "Yes"]
        )

    st.divider()

    # -----------------------------------------------------
    # MEDICAL HISTORY
    # -----------------------------------------------------

    st.subheader(
        "Medical History"
    )

    col1, col2 = st.columns(2)

    with col1:

        heart_disease = st.selectbox(
            "Heart Disease",
            ["No", "Yes"]
        )

    with col2:

        smoking_values = (
            data["smoking_history"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        smoking_history = st.selectbox(
            "Smoking History",
            smoking_values
        )

    st.write("")

    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    if st.button(
        "Generate Prediction",
        type="primary",
        use_container_width=True
    ):

        if patient_id.strip() == "":

            st.warning(
                "Please enter a Patient ID."
            )

        else:

            try:

                result = predict_diabetes(

                    patient_id.strip(),

                    gender,

                    age,

                    1 if hypertension == "Yes"
                    else 0,

                    1 if heart_disease == "Yes"
                    else 0,

                    smoking_history,

                    bmi,

                    blood_glucose

                )

                st.session_state.history.append(
                    result
                )

                st.success(
                    "Prediction completed successfully."
                )

                st.divider()

                st.subheader(
                    "Prediction Result"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Predicted Probability",
                        f"{result['Probability']:.2f}%"
                    )

                with col2:

                    st.metric(
                        "Risk Category",
                        result["Risk"]
                    )

                st.progress(
                    int(
                        min(
                            result["Probability"],
                            100
                        )
                    )
                )

                if result["Risk"] == "High Risk":

                    st.error(
                        "High Risk"
                    )

                elif result["Risk"] == "Moderate Risk":

                    st.warning(
                        "Moderate Risk"
                    )

                else:

                    st.success(
                        "Low Risk"
                    )

                st.caption(
                    "This result is an AI-generated prediction "
                    "for academic demonstration and is not a "
                    "medical diagnosis."
                )

            except Exception as e:

                st.error(
                    "Prediction failed."
                )

                st.write(
                    "Please check the model and dataset "
                    "configuration."
                )

                st.write(
                    str(e)
                )


# =========================================================
# HISTORY
# =========================================================

elif page == "Assessment History":

    st.title(
        "Assessment History"
    )

    st.write(
        "Review previous assessment records."
    )

    st.divider()

    if len(st.session_state.history) == 0:

        st.info(
            "No assessment records available."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        col1, col2 = st.columns(2)

        with col1:

            search = st.text_input(
                "Search Patient ID"
            )

        with col2:

            risk_filter = st.selectbox(
                "Risk Level",
                [
                    "All",
                    "High Risk",
                    "Moderate Risk",
                    "Low Risk"
                ]
            )

        filtered = history_df.copy()

        if search:

            filtered = filtered[
                filtered["Patient ID"]
                .astype(str)
                .str.contains(
                    search,
                    case=False,
                    na=False
                )
            ]

        if risk_filter != "All":

            filtered = filtered[
                filtered["Risk"]
                == risk_filter
            ]

        display = filtered.copy()

        display["Probability"] = (
            display["Probability"]
            .round(2)
            .astype(str)
            + "%"
        )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# ANALYTICS
# =========================================================

elif page == "Risk Analytics":

    st.title(
        "Risk Analytics"
    )

    st.write(
        "Overview of prediction results."
    )

    st.divider()

    if len(st.session_state.history) == 0:

        st.info(
            "No data available."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        risk_counts = (
            history_df["Risk"]
            .value_counts()
            .reindex(
                [
                    "High Risk",
                    "Moderate Risk",
                    "Low Risk"
                ],
                fill_value=0
            )
        )

        st.subheader(
            "Risk Distribution"
        )

        st.bar_chart(
            risk_counts
        )

        st.divider()

        st.subheader(
            "Clinical Overview"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Average Age",
                f"{history_df['Age'].mean():.1f}"
            )

        with col2:

            st.metric(
                "Average BMI",
                f"{history_df['BMI'].mean():.1f}"
            )

        with col3:

            st.metric(
                "Average Glucose",
                f"{history_df['Blood Glucose'].mean():.1f}"
            )


# =========================================================
# AI MODEL
# =========================================================

elif page == "AI Model":

    st.title(
        "AI Model"
    )

    st.write(
        "Machine learning model information."
    )

    st.divider()

    st.subheader(
        "✦ Diabetes Risk Prediction Model"
    )

    st.metric(
    "Model Accuracy",
    f"{model_accuracy * 100:.2f}%"
    )

    st.write(
        "The system uses a trained machine learning "
        "classification model to generate a predicted "
        "probability of diabetes."
    )

    st.divider()

    st.subheader(
        "Model Features"
    )

    features = pd.DataFrame({
        "Feature": model_features
    })

    st.dataframe(
        features,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader(
        "Risk Categories"
    )

    st.write(
        "High Risk — predicted probability ≥ 70%"
    )

    st.write(
        "Moderate Risk — predicted probability 40%–69.99%"
    )

    st.write(
        "Low Risk — predicted probability < 40%"
    )

    st.caption(
        "These thresholds are application-defined "
        "for academic demonstration and are not "
        "clinically validated."
    )


# =========================================================
# ABOUT
# =========================================================

elif page == "About":

    st.title(
        "About DiabetesAI Hospital"
    )

    st.write(
        "DiabetesAI Hospital is an academic prototype "
        "that demonstrates the integration of machine "
        "learning into a clinical risk assessment system."
    )

    st.divider()

    st.subheader(
        "Purpose"
    )

    st.write(
        "The application accepts selected patient "
        "information and uses a trained machine learning "
        "model to generate a diabetes risk prediction."
    )

    st.subheader(
        "Main Features"
    )

    st.write(
        "• New patient assessment"
    )

    st.write(
        "• AI risk prediction"
    )

    st.write(
        "• Assessment history"
    )

    st.write(
        "• Risk analytics"
    )

    st.write(
        "• AI model information"
    )

    st.divider()

    st.subheader(
        "Important Notice"
    )

    st.warning(
        "This is an academic prototype. The prediction "
        "is not a medical diagnosis and should not be "
        "used to make healthcare decisions."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "DiabetesAI Hospital · Clinical Risk Prediction System · Academic Prototype"
)