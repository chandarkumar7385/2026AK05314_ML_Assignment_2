
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Application configuration

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Bank Marketing Term Deposit Prediction")
st.write(
    "Upload a test dataset, select a trained classification model "
    "and evaluate its performance."
)

# Model directory and available models

MODEL_DIR = Path("model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbours": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Linear SVM": "linear_svm.joblib"
}

# Upload test dataset

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

# Select classification model

st.header("2. Select Classification Model")

selected_model = st.selectbox(
    "Choose a trained model",
    list(MODEL_FILES.keys())
)

# Evaluate the uploaded dataset

if uploaded_file is not None:

    try:
        test_df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(test_df.head())

        if "y" not in test_df.columns:
            st.error("The uploaded CSV must contain the target column 'y'.")

        else:
            X_test = test_df.drop(columns=["y"])
            y_test = test_df["y"]

            expected_features = [
                "age", "job", "marital", "education", "default",
                "balance", "housing", "loan", "contact", "day",
                "month", "campaign", "pdays", "previous", "poutcome"
            ]

            if set(X_test.columns) != set(expected_features):
                st.error(
                    "The uploaded CSV must contain all 15 expected "
                    "input features, without missing or extra columns."
                )

            else:
                # Load the selected saved model

                model_path = MODEL_DIR / MODEL_FILES[selected_model]

                if not model_path.exists():
                    st.error(f"Model file not found: {model_path}")

                else:
                    model = joblib.load(model_path)

                    # Generate predictions

                    y_pred = model.predict(X_test)

                    # Calculate probability or decision scores for AUC

                    if hasattr(model, "predict_proba"):
                        scores = model.predict_proba(X_test)[:, 1]
                    else:
                        scores = model.decision_function(X_test)

                    # Calculate evaluation metrics

                    metrics = {
                        "Accuracy": accuracy_score(y_test, y_pred),
                        "AUC": roc_auc_score(y_test, scores),
                        "Precision": precision_score(
                            y_test, y_pred, pos_label="yes",
                            zero_division=0
                        ),
                        "Recall": recall_score(
                            y_test, y_pred, pos_label="yes",
                            zero_division=0
                        ),
                        "F1 Score": f1_score(
                            y_test, y_pred, pos_label="yes",
                            zero_division=0
                        ),
                        "MCC": matthews_corrcoef(y_test, y_pred)
                    }

                    st.success(
                        f"Evaluation completed using {selected_model}."
                    )

                    # Display evaluation metrics

                    st.header("3. Model Evaluation Metrics")

                    metric_columns = st.columns(3)

                    for index, (name, value) in enumerate(metrics.items()):
                        metric_columns[index % 3].metric(
                            name, f"{value:.4f}"
                        )

                    # Display confusion matrix

                    st.header("4. Confusion Matrix")

                    cm = confusion_matrix(
                        y_test, y_pred, labels=["no", "yes"]
                    )

                    fig, ax = plt.subplots(figsize=(6, 4))

                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=["no", "yes"],
                        yticklabels=["no", "yes"],
                        ax=ax
                    )

                    ax.set_xlabel("Predicted Label")
                    ax.set_ylabel("Actual Label")
                    ax.set_title(f"{selected_model} - Confusion Matrix")

                    st.pyplot(fig)
                    plt.close(fig)

                    # Display classification report

                    st.header("5. Classification Report")

                    report = classification_report(
                        y_test,
                        y_pred,
                        labels=["no", "yes"],
                        zero_division=0,
                        output_dict=True
                    )

                    st.dataframe(
                        pd.DataFrame(report).transpose().round(4)
                    )

    except Exception as error:
        st.error(f"Unable to evaluate the dataset: {error}")

else:
    st.info("Please upload a test CSV file to begin evaluation.")
