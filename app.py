import streamlit as st
import pandas as pd
import joblib

from pathlib import Path


# ==================================================
# 1. PAGE SETTINGS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "diabetes_model.pkl"
CURVE_PATH = BASE_DIR / "calibration_curve.png"

st.set_page_config(
    page_title="DiabetesAI Hospital",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==================================================
# 2. CUSTOM CSS
# ==================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #163a5f;
    --navy-deep: #0d2b49;
    --teal: #18a999;
    --teal-dark: #118578;
    --teal-soft: #e7f7f4;
    --blue-soft: #edf4f9;
    --background: #f7f9fc;
    --white: #ffffff;
    --text: #18334d;
    --muted: #607487;
    --border: #dfe7ed;
}

.stApp {
    background: var(--background);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1180px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: var(--navy) !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: -0.5px;
}

p, label, li {
    font-family: 'Inter', sans-serif;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--border);
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 4px 16px rgba(22, 58, 95, 0.04);
}

[data-testid="stMetricLabel"] {
    color: var(--muted);
}

[data-testid="stMetricValue"] {
    color: var(--navy);
}

div.stButton > button,
div.stFormSubmitButton > button {
    border-radius: 10px;
    min-height: 44px;
    font-weight: 600;
    transition: all 0.2s ease;
}

div.stButton > button[kind="primary"],
div.stFormSubmitButton > button[kind="primary"] {
    background: var(--teal);
    color: white;
    border: 1px solid var(--teal);
}

div.stButton > button[kind="primary"]:hover,
div.stFormSubmitButton > button[kind="primary"]:hover {
    background: var(--teal-dark);
    border-color: var(--teal-dark);
}

div.stButton > button[kind="secondary"] {
    border: 1px solid var(--border);
    color: var(--navy);
    background: white;
}

div.stButton > button[kind="secondary"]:hover {
    border-color: var(--teal);
    color: var(--teal-dark);
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 10px;
}

.hero {
    background: linear-gradient(
        125deg, #0d2b49 0%, #163a5f 65%, #176d79 100%
    );
    color: white;
    border-radius: 24px;
    padding: 42px 38px;
    margin: 10px 0 28px 0;
    box-shadow: 0 12px 35px rgba(13, 43, 73, 0.12);
}

.hero h1 {
    color: white !important;
    font-size: clamp(2rem, 5vw, 3.1rem);
    line-height: 1.15;
    margin-bottom: 16px;
}

.hero p {
    color: #e0edf5;
    font-size: 1.05rem;
    line-height: 1.8;
}

.hero-tag {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    color: #bff7ed;
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 30px;
    padding: 7px 13px;
    font-size: 0.85rem;
    margin-bottom: 18px;
}

.section-label {
    color: var(--teal-dark);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-size: 0.78rem;
    font-weight: 700;
    margin-bottom: 8px;
}

.feature-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 24px;
    min-height: 205px;
    box-shadow: 0 4px 18px rgba(22, 58, 95, 0.035);
}

.feature-icon {
    background: var(--teal-soft);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 48px;
    height: 48px;
    border-radius: 13px;
    font-size: 1.5rem;
    margin-bottom: 14px;
}

.feature-card h3 {
    font-size: 1.1rem;
    margin: 4px 0 10px 0;
}

.feature-card p {
    color: var(--muted);
    line-height: 1.7;
    font-size: 0.94rem;
}

.info-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 16px;
}

.result-positive {
    background: #fff6e8;
    border: 1px solid #f1d5a5;
    border-radius: 18px;
    padding: 24px;
    margin: 14px 0;
}

.result-negative {
    background: var(--teal-soft);
    border: 1px solid #b6e6dc;
    border-radius: 18px;
    padding: 24px;
    margin: 14px 0;
}

.result-positive h3 {
    color: #885514 !important;
}

.result-negative h3 {
    color: var(--teal-dark) !important;
}

.small-note {
    color: var(--muted);
    font-size: 0.86rem;
    line-height: 1.7;
}

.footer {
    text-align: center;
    color: var(--muted);
    border-top: 1px solid var(--border);
    padding-top: 22px;
    margin-top: 38px;
    font-size: 0.85rem;
}

div[data-testid="stProgress"] > div > div {
    background-color: var(--teal);
}

@media (max-width: 640px) {
    .block-container {
        padding: 1rem 1rem 2rem 1rem;
    }

    .hero {
        padding: 28px 22px;
        border-radius: 18px;
    }

    .feature-card, .info-card {
        padding: 18px;
    }
}

/* Fix Assessment form label visibility */
.stApp label,
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stWidgetLabel"] label,
.stApp .stSelectbox label,
.stApp .stNumberInput label {
    color: #163A5F !important;
    opacity: 1 !important;
}


/* Make input text softer */
.stApp input,
.stApp textarea,
.stApp [data-baseweb="select"] {
    color: #526579 !important;
}

/* Keep dropdown options readable */
.stApp [role="option"] {
    color: #526579 !important;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. SESSION STATE
# ==================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "assessment_result" not in st.session_state:
    st.session_state.assessment_result = None

if "assessment_input" not in st.session_state:
    st.session_state.assessment_input = None


def go_to(page):
    st.session_state.page = page


# ==================================================
# 4. HEADER AND NAVIGATION
# ==================================================

def show_header():
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;
                padding:8px 0 16px 0;">
        <div style="background:#e7f7f4;border-radius:13px;
                    padding:10px;font-size:26px;">🩺</div>
        <div>
            <div style="font-size:1.2rem;font-weight:800;
                        color:#163a5f;">DiabetesAI Hospital</div>
            <div style="font-size:0.8rem;color:#607487;">
                AI-assisted health education prototype
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav1, nav2, nav3, nav4, nav5 = st.columns(5)

    navigation = [
        (nav1, "Home"),
        (nav2, "Assessment"),
        (nav3, "Dietary Guidance"),
        (nav4, "Prevention Guide"),
        (nav5, "Model Insights")
    ]

    for column, page_name in navigation:
        with column:
            if st.button(
                page_name,
                key=f"nav_{page_name}",
                type=(
                    "primary"
                    if st.session_state.page == page_name
                    else "secondary"
                ),
                use_container_width=True
            ):
                go_to(page_name)
                st.rerun()

    st.divider()


# ==================================================
# 5. FOOTER
# ==================================================

def show_footer():
    st.markdown("""
    <div class="footer">
        <strong>DiabetesAI Hospital</strong><br>
        An educational prototype for learning about diabetes-related
        health information and machine learning.<br><br>
        This application does not provide medical diagnosis or treatment.
    </div>
    """, unsafe_allow_html=True)


# ==================================================
# 6. LOAD TRAINED MODEL
# ==================================================

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("Model not found. Please run train_model.py first.")
    st.stop()

try:
    bundle = load_bundle()
    model = bundle["model"]
    features = bundle["feature_columns"]

except Exception as error:
    st.error("Could not load the trained model.")
    st.exception(error)
    st.stop()


# ==================================================
# 7. HOME PAGE
# ==================================================

def show_home():

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">✦ HEALTH TECHNOLOGY & EDUCATION</div>
        <h1>Understand your health.<br>Build healthier habits.</h1>
        <p>
            Explore diabetes-related health information through an
            educational machine learning prototype. Enter your
            information, view the model's classification, and discover
            practical healthy-living guidance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-label">GET STARTED</div>',
        unsafe_allow_html=True
    )

    st.header("Your health journey starts here")

    st.write(
        "Use the assessment to explore how a trained model classifies "
        "the information you provide. You can then explore educational "
        "content about healthy eating and prevention."
    )

    if st.button(
        "Start Health Assessment →",
        type="primary",
        use_container_width=True
    ):
        go_to("Assessment")
        st.rerun()

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📋</div>
            <h3>Health Assessment</h3>
            <p>
                Enter basic health information and explore the output
                of the trained machine learning model.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🥗</div>
            <h3>Healthy Eating</h3>
            <p>
                Learn about balanced meals, nutritious food choices,
                and everyday eating habits.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">💚</div>
            <h3>Prevention Education</h3>
            <p>
                Explore healthy lifestyle habits that support
                general wellbeing and diabetes prevention.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    <div class="info-card">
        <h3>Important health notice</h3>
        <p>
            This prototype is for educational purposes only.
            Its predictions are not a medical diagnosis and should
            not be used to make treatment decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)



# ==================================================
# 8. ASSESSMENT PAGE
# ==================================================

def show_assessment():

    st.markdown(
        '<div class="section-label">PERSONAL HEALTH INFORMATION</div>',
        unsafe_allow_html=True
    )

    st.title("Diabetes Pattern Assessment")

    st.write(
        "Enter the information below to explore the classification "
        "produced by the machine learning model."
    )

    st.info(
        "This assessment is an educational prototype, not a "
        "medical diagnostic tool."
    )

    with st.form("assessment_form"):

        st.subheader("Basic Information")

        col1, col2 = st.columns(2)

        user_data = {}

        with col1:
            if "gender" in features:
                user_data["gender"] = st.selectbox(
                    "Gender",
                    ["Female", "Male", "Other"]
                )

            if "age" in features:
                user_data["age"] = st.number_input(
                    "Age (years)",
                    min_value=1.0,
                    max_value=100.0,
                    value=40.0,
                    step=1.0
                )

        with col2:
            if "smoking_history" in features:
                user_data["smoking_history"] = st.selectbox(
                    "Smoking History",
                    [
                        "No Info",
                        "never",
                        "former",
                        "current",
                        "not current",
                        "ever"
                    ]
                )

        st.divider()
        st.subheader("Health Information")

        col3, col4 = st.columns(2)

        with col3:
            if "bmi" in features:
                user_data["bmi"] = st.number_input(
                    "BMI (kg/m²)",
                    min_value=10.0,
                    max_value=80.0,
                    value=27.0,
                    step=0.1
                )

            if "HbA1c_level" in features:
                user_data["HbA1c_level"] = st.number_input(
                    "HbA1c Level (%)",
                    min_value=3.0,
                    max_value=20.0,
                    value=5.7,
                    step=0.1
                )

        with col4:
            if "blood_glucose_level" in features:
                user_data["blood_glucose_level"] = st.number_input(
                    "Blood Glucose (mg/dL)",
                    min_value=40.0,
                    max_value=500.0,
                    value=100.0,
                    step=1.0
                )

            if "hypertension" in features:
                user_data["hypertension"] = st.selectbox(
                    "Hypertension",
                    [0, 1],
                    format_func=lambda value:
                    "Yes" if value == 1 else "No"
                )

            if "heart_disease" in features:
                user_data["heart_disease"] = st.selectbox(
                    "Heart Disease",
                    [0, 1],
                    format_func=lambda value:
                    "Yes" if value == 1 else "No"
                )

        submitted = st.form_submit_button(
            "Assess My Information",
            type="primary",
            use_container_width=True
        )

    if submitted:

        # Make sure every model feature has an input
        missing_features = [
            feature for feature in features
            if feature not in user_data
        ]

        if missing_features:
            st.error(
                "Some model features are not supported by the form: "
                + ", ".join(missing_features)
            )
            return

        # Keep exactly the same feature order as model training
        input_df = pd.DataFrame(
            [[user_data[feature] for feature in features]],
            columns=features
        )

        try:
            prediction = int(model.predict(input_df)[0])

            classes = list(model.classes_)
            positive_index = classes.index(1)

            probability = float(
                model.predict_proba(input_df)[0][positive_index]
            )

            # Check values against the training data ranges
            warnings = []
            ranges = bundle.get("training_ranges", {})

            for feature, limits in ranges.items():
                if feature in user_data:
                    value = float(user_data[feature])

                    if value < limits[0] or value > limits[1]:
                        warnings.append(
                            f"{feature}: entered value is outside "
                            f"the training data range "
                            f"({limits[0]:.2f}–{limits[1]:.2f})."
                        )

            # Save results so they remain available on other pages
            st.session_state.assessment_result = {
                "prediction": prediction,
                "probability": probability,
                "warnings": warnings
            }

            st.session_state.assessment_input = user_data.copy()

            # Automatically navigate to the corresponding page
            if prediction == 1:
                go_to("Dietary Guidance")
            else:
                go_to("Prevention Guide")

            st.rerun()

        except Exception as error:
            st.error("Prediction failed.")
            st.code(str(error))

    st.write("")

    st.markdown("""
    <div class="info-card">
        <h3>Before you begin</h3>
        <p>
            Use accurate information where available. If you do not
            know a medical measurement, consult a qualified healthcare
            professional rather than guessing.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ==================================================
# 9. DISPLAY THE SAVED ASSESSMENT RESULT
# ==================================================

def show_result_summary():

    result = st.session_state.assessment_result

    if result is None:
        st.info(
            "You have not completed an assessment yet. "
            "Start an assessment to view a model result."
        )

        if st.button(
            "Start Assessment",
            type="primary",
            use_container_width=True
        ):
            go_to("Assessment")
            st.rerun()

        return False

    prediction = result["prediction"]
    probability = result["probability"]

    if prediction == 1:

        st.markdown("""
        <div class="result-positive">
            <h3>Model Classification: Positive</h3>
            <p>
                The model classified the submitted information as
                belonging to its diabetes-positive class.
                This is not a confirmed diagnosis.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="result-negative">
            <h3>Model Classification: Negative</h3>
            <p>
                The model classified the submitted information as
                belonging to its diabetes-negative class.
                This does not rule out diabetes.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.metric(
        "Model Positive-Class Probability",
        f"{probability * 100:.1f}%"
    )

    st.progress(probability)

    st.caption(
        "This value is the model's estimated probability for its "
        "positive class. It is not a clinically validated estimate "
        "of your personal probability of having diabetes."
    )

    if result["warnings"]:
        st.warning(
            "Some values are outside the ranges observed in the "
            "training data:\n\n- "
            + "\n- ".join(result["warnings"])
        )

    with st.expander("View submitted information"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Feature": key,
                        "Value": value
                    }
                    for key, value
                    in st.session_state.assessment_input.items()
                ]
            ),
            hide_index=True,
            use_container_width=True
        )

    st.warning(
        "This result is for educational purposes only. "
        "Consult a qualified healthcare professional for assessment."
    )

    return True


# ==================================================
# 10. DIETARY GUIDANCE PAGE
# ==================================================

def show_dietary():

    st.markdown(
        '<div class="section-label">HEALTHY EATING</div>',
        unsafe_allow_html=True
    )

    st.title("Dietary Guidance")

    st.write(
        "Explore general healthy-eating ideas that may support "
        "overall health. These suggestions are educational and "
        "are not a personalised medical diet."
    )

    has_result = show_result_summary()

    if not has_result:
        return

    st.divider()
    st.subheader("Build a balanced plate")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🥦</div>
            <h3>Vegetables</h3>
            <p>
                Include a variety of vegetables, such as broccoli,
                spinach, cabbage, cucumber and leafy greens.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🌾</div>
            <h3>Whole Grains</h3>
            <p>
                Consider oats, wholemeal bread, brown rice or other
                fibre-rich carbohydrate sources in balanced portions.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🐟</div>
            <h3>Protein</h3>
            <p>
                Choose suitable protein sources such as fish, eggs,
                tofu, beans or lean meat.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    <div class="info-card">
        <h3>Everyday food choices</h3>
        <ul>
            <li>Drink water regularly and limit sugary drinks.</li>
            <li>Choose a variety of minimally processed foods.</li>
            <li>Pay attention to portion sizes and meal balance.</li>
            <li>Include fibre-rich foods as part of balanced meals.</li>
            <li>Keep regular meal routines that suit your needs.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3>Foods to limit</h3>
        <ul>
            <li>Frequent sugary drinks and sweetened beverages.</li>
            <li>Excessive amounts of sweets and desserts.</li>
            <li>Frequent consumption of highly processed foods.</li>
        </ul>
        <p>
            You do not need to make extreme dietary changes based
            on this model result.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "If you have diabetes, take medication, are pregnant, or have "
        "specific dietary needs, seek guidance from a doctor or "
        "registered dietitian before changing your diet."
    )

    if st.button(
        "Back to Home",
        use_container_width=True
    ):
        go_to("Home")
        st.rerun()


# ==================================================
# 11. PREVENTION GUIDE PAGE
# ==================================================

def show_prevention():

    st.markdown(
        '<div class="section-label">HEALTHY LIVING</div>',
        unsafe_allow_html=True
    )

    st.title("Prevention Guide")

    st.write(
        "Learn about everyday habits that support general wellbeing "
        "and may help reduce the risk of type 2 diabetes."
    )

    has_result = show_result_summary()

    if not has_result:
        return

    st.divider()
    st.subheader("Healthy habits to explore")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🏃</div>
            <h3>Stay Active</h3>
            <p>
                Find enjoyable ways to move, such as walking,
                cycling, swimming or other suitable activities.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🥗</div>
            <h3>Eat Balanced Meals</h3>
            <p>
                Include vegetables, suitable protein sources,
                fibre-rich foods and balanced portions.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">😴</div>
            <h3>Get Enough Sleep</h3>
            <p>
                Maintain a regular sleep routine and aim for
                sufficient rest based on your age and needs.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🩺</div>
            <h3>Health Check-ups</h3>
            <p>
                Discuss appropriate screening and follow-up with
                a qualified healthcare professional.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.subheader("My Healthy Habits Checklist")

    st.write(
        "Tick the habits you would like to work on. "
        "This checklist is for your current session."
    )

    st.checkbox(
        "I will choose water more often instead of sugary drinks.",
        key="habit_water"
    )

    st.checkbox(
        "I will include vegetables in my meals.",
        key="habit_vegetables"
    )

    st.checkbox(
        "I will make time for suitable physical activity.",
        key="habit_activity"
    )

    st.checkbox(
        "I will try to maintain a regular sleep schedule.",
        key="habit_sleep"
    )

    st.checkbox(
        "I will seek professional advice when needed.",
        key="habit_checkup"
    )

    st.info(
        "A negative model classification does not guarantee that "
        "you are free from diabetes. Appropriate screening depends "
        "on your circumstances and professional advice."
    )

    if st.button(
        "Back to Home",
        use_container_width=True
    ):
        go_to("Home")
        st.rerun()


# ==================================================
# 12. MODEL INSIGHTS PAGE
# ==================================================

def show_insights():

    st.markdown(
        '<div class="section-label">MACHINE LEARNING</div>',
        unsafe_allow_html=True
    )

    st.title("Model Insights")

    st.write(
        "Explore how the trained machine learning model performed "
        "on the held-out test dataset."
    )

    st.markdown("""
    <div class="info-card">
        <h3>About the model</h3>
        <p>
            This project uses a Random Forest classification pipeline.
            Categorical features are processed using One-Hot Encoding,
            while numerical features are passed to the model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Evaluation Metrics")

    metric_names = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1 Score")
    ]

    columns = st.columns(4)

    for column, (key, label) in zip(columns, metric_names):
        with column:
            value = bundle.get(key)

            if value is not None:
                st.metric(label, f"{float(value):.3f}")
            else:
                st.metric(label, "Unavailable")

    st.caption(
        "These metrics were saved during model training using the "
        "held-out test split. They do not establish clinical validity."
    )

    st.divider()
    st.subheader("Probability Calibration")

    brier = bundle.get("brier_score")

    if brier is not None:
        st.metric("Brier Score", f"{float(brier):.4f}")

        st.caption(
            "The Brier score measures the difference between "
            "predicted probabilities and actual labels. Lower values "
            "are generally better when comparing models on the same "
            "dataset and evaluation setup."
        )

    if CURVE_PATH.exists():
        st.image(
            str(CURVE_PATH),
            caption="Calibration curve generated during model training",
            use_container_width=True
        )
    else:
        st.info(
            "Calibration curve not found. Run train_model.py to "
            "generate it."
        )

    st.warning(
        "Model evaluation metrics describe performance on the test "
        "dataset. They do not guarantee accuracy for an individual "
        "user or establish that the model is suitable for clinical use."
    )


# ==================================================
# 13. MAIN APPLICATION
# ==================================================

show_header()

if st.session_state.page == "Home":
    show_home()

elif st.session_state.page == "Assessment":
    show_assessment()

elif st.session_state.page == "Dietary Guidance":
    show_dietary()

elif st.session_state.page == "Prevention Guide":
    show_prevention()

elif st.session_state.page == "Model Insights":
    show_insights()

else:
    go_to("Home")
    st.rerun()

show_footer()
