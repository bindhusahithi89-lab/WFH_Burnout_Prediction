import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px





# --------------------------
# Streamlit Gradient Background
# -------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #1f3c88, #6a1b9a);
        color: white;
    }

    /* Button style */
    .stButton>button {
        background-color: #ff6f61;
        color: white;
        font-size: 16px;
    }

    /* Slider track and handle */
    .stSlider > div[data-baseweb="slider"] > div {
        background-color: #4db6ac !important;
    }

    /* Labels for sliders, selectboxes, and numeric inputs */
    .css-1kyxreq, /* selectbox label */
    .css-1v0mbdj, /* slider label */
    .css-14xtw13, /* numeric input label */
    label {
        color: white !important;
        font-weight: bold;
    }

    /* Slider range numbers (min, max, current value) */
    .css-10trblm, /* current value */
    .css-1n76uvr, /* min value */
    .css-1q7s65k { /* max value */
        color: white !important;
        font-weight: bold;
    }
    /* ✅ Classification report styling */
    .classification-report {
        background-color: rgba(0, 0, 0, 0.25);
        padding: 15px;
        border-radius: 10px;
        font-family: monospace;
        white-space: pre;
        color: white;
        line-height: 1.5;
        border: 1px solid rgba(255,255,255,0.2);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------
# Load Dataset
# --------------------------
source_df = pd.read_csv("work_from_home_burnout_dataset.csv")
# --------------------------
# Clean dataset
# --------------------------

# 1. Strip whitespace from column names (safety)
source_df.columns = source_df.columns.str.strip()

print(source_df.columns)

# 2. Encode 'day_type' (Weekday/Weekend)
# Encode 'day_type' in SAME position
le_day = LabelEncoder()
day_type_index = list(source_df.columns).index('day_type')

source_df.insert(day_type_index, 'day_type_encoded',
          le_day.fit_transform(source_df['day_type']))

# 3. Encode target 'burnout_risk'
le_target = LabelEncoder()
source_df['burnout_risk_encoded'] = le_target.fit_transform(source_df['burnout_risk'])

# 4. Drop original categorical columns
source_df = source_df.drop(['day_type', 'burnout_risk','user_id'], axis=1)

# --------------------------
# Features and target
# --------------------------
X = source_df.drop(['burnout_risk_encoded'], axis=1)
y = source_df['burnout_risk_encoded']

# Balance classes using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train Random Forest

model = RandomForestClassifier(n_estimators=200, max_features='sqrt', class_weight='balanced',random_state=42)
model.fit(X_train, y_train)

# --------------------------
# Evaluate Model
# --------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=le_target.classes_)


# Show model evaluation metrics
st.subheader("Model Performance on Test Data")
st.write(f"**Accuracy:** {acc:.2f}")

st.write("**Confusion Matrix:**")
fig, ax = plt.subplots()
sns.heatmap(conf_mat, annot=True, fmt="d", xticklabels=le_target.classes_, yticklabels=le_target.classes_, cmap="Blues")
plt.ylabel('Actual')
plt.xlabel('Predicted')
st.pyplot(fig)
plt.close(fig)

st.write("**Classification Report:**")

st.code(class_report)

# --------------------------
# Streamlit UI
# --------------------------
st.title("💻 WFH Burnout Risk Prediction")
st.write("Input employee work data to predict burnout risk:")

# user_id = st.number_input("User ID", min_value=1, max_value=9999, value=1)
day_type = st.selectbox("Day Type", ("Weekend", "Weekday"))
work_hours = st.slider("Work Hours", 0, 24, 8)
screen_time_hours = st.slider("Screen Time Hours", 0, 24, 8)
meetings_count = st.slider("Number of Meetings", 0, 20, 3)
breaks_taken = st.slider("Breaks Taken", 0, 10, 2)
after_hours_work = st.slider("After Hours Work", 0, 24, 8)
sleep_hours = st.slider("Sleep Hours", 0, 12, 7)
task_completion_rate = st.slider("Task Completion Rate (%)", 0, 100, 80)
burnout_score = st.slider("Burnout Score", 0, 150, 50)


# Convert inputs to model format
day_type_val = 1 if day_type == "Weekday" else 0
after_hours_val = 1 if after_hours_work > 0 else 0

print(X.columns)
# Make sure columns match exactly
input_source_df = pd.DataFrame([[ 
    day_type_val, work_hours, screen_time_hours, meetings_count,
    breaks_taken, after_hours_val, sleep_hours, task_completion_rate,burnout_score
]], columns=X.columns)

# Predict
if st.button("Predict Burnout Risk"):
    pred = model.predict(input_source_df)
    st.success(f"Predicted Burnout Risk: {le_target.inverse_transform(pred)[0]}")




st.subheader("📊 Interactive Data Visualizations")

# Create readable labels (IMPORTANT)
source_df['burnout_label'] = source_df['burnout_risk_encoded'].map(
    dict(enumerate(le_target.classes_))
)

# Sidebar option
vis_option = st.sidebar.selectbox("Select Visualization Type", [
    "Burnout Risk Distribution",
    "Feature Correlation Heatmap",
    "Feature Importance",
    "Histogram of Feature",
    "Boxplot by Burnout Risk",
    "Scatter Matrix (Feature Relationships)"
])

# --------------------------
# 1. Burnout Risk Distribution
# --------------------------
if vis_option == "Burnout Risk Distribution":
    fig1, ax1 = plt.subplots()

    sns.countplot(
        data=source_df,
        x='burnout_label',
        hue='burnout_label',   # ✅ add hue
        palette="viridis",
        order=source_df['burnout_label'].value_counts().index,
        legend=False  # ✅ removes duplicate legend
    )

    ax1.set_title("Burnout Risk Distribution")
    ax1.set_xlabel("Burnout Risk")
    ax1.set_ylabel("Count")

    st.pyplot(fig1)
    plt.close(fig1)

# --------------------------
# 2. Correlation Heatmap
# --------------------------
elif vis_option == "Feature Correlation Heatmap":
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    numeric_source_df = source_df.drop(['burnout_risk_encoded', 'burnout_label'], axis=1)

    sns.heatmap(numeric_source_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")

    plt.title("Feature Correlations")

    st.pyplot(fig2)
    plt.close(fig2)

# --------------------------
# 3. Feature Importance
# --------------------------
elif vis_option == "Feature Importance":
    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    fig3, ax3 = plt.subplots()

    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="magma")

    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    st.pyplot(fig3)
    plt.close(fig3)

# --------------------------
# 4. Histogram
# --------------------------
elif vis_option == "Histogram of Feature":
    feature = st.sidebar.selectbox("Select Feature", X.columns)

    fig4, ax4 = plt.subplots()

    sns.histplot(source_df[feature], kde=True, color="teal")

    plt.title(f"Histogram of {feature}")

    st.pyplot(fig4)
    plt.close(fig4)

# --------------------------
# 5. Boxplot
# --------------------------
elif vis_option == "Boxplot by Burnout Risk":
    feature = st.sidebar.selectbox("Select Feature", X.columns)

    fig5, ax5 = plt.subplots()

    sns.boxplot(
        x='burnout_label',
        y=feature,
        data=source_df,
        palette="Set2"
    )

    plt.title(f"{feature} by Burnout Risk")

    st.pyplot(fig5)
    plt.close(fig5)

# --------------------------
# 6. Scatter Matrix
# --------------------------
elif vis_option == "Scatter Matrix (Feature Relationships)":

    fig6 = px.scatter_matrix(
        source_df,
        dimensions=X.columns,
        color="burnout_label",
        title="Feature Relationships"
    )

    st.plotly_chart(fig6, use_container_width=True)