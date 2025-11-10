import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# --- Streamlit Page Config ---
st.set_page_config(page_title="ML Model Builder", layout="wide")

st.title("Machine Learning Model Builder & Visualizer ")
st.caption("Upload a supervised ML dataset (CSV file) — it must include one target column (output) and several feature columns (inputs like age, salary, gender, etc.).")

# --- File uploader ---
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    st.write(f"**Dataset shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    # --- Target selection ---
    target_column = st.selectbox("Select the target column", df.columns)

    # --- Features and target ---
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle categorical data
    X = pd.get_dummies(X, drop_first=True)
    
    # --- Handle Missing Values ---
    st.subheader("Handling Missing Values")

    missing_before = df.isnull().sum().sum()
    st.write(f"Total missing values before cleaning: {missing_before}")

    # Separate numeric and categorical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    # Fill numeric columns with mean
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    # Fill categorical columns with mode
    for col in cat_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    missing_after = df.isnull().sum().sum()
    st.success(f"Missing values handled. Remaining missing: {missing_after}")


    # --- Data split ---
    test_size = st.slider("Test size (as fraction)", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # --- Model selection ---
    st.subheader("Model Configuration")
    model_name = st.selectbox("Choose a model", ["Logistic Regression", "Random Forest"])

    if model_name == "Logistic Regression":
        C = st.slider("Inverse regularization strength (C)", 0.01, 10.0, 1.0)
        model = LogisticRegression(C=C, max_iter=1000)
    else:
        n_estimators = st.slider("Number of trees", 10, 300, 100)
        max_depth = st.slider("Max depth", 1, 30, 5)
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )

    # --- Train the model ---
    if st.button("Train Model"):
        if model_name == "Logistic Regression":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # --- Results ---
        st.success(f"Model trained successfully!")
        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")

        # --- Classification Report ---
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        # --- Confusion Matrix ---
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        
else:
    st.info("Upload a CSV file to begin.")
