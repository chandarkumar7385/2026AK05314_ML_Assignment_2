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
    "Upload a test dataset, compare five trained classification "
    "models and evaluate individual model performance."
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

EXPECTED_FEATURES = [
    "age", "job", "marital", "education", "default",
    "balance", "housing", "loan", "contact", "day",
    "month", "campaign", "pdays", "previous", "poutcome"
]


# Function to calculate model metrics

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, scores),
        "Precision": precision_score(
            y_test, y_pred, pos_label="yes", zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, pos_label="yes", zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, y_pred, pos_label="yes", zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    return metrics, y_pred


# Upload test dataset

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    try:
        test_df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(test_df.head())

        if "y" not in test_df.columns:
            st.error("The uploaded CSV must contain target column 'y'.")

        elif set(test_df.drop(columns=["y"]).columns) != set(EXPECTED_FEATURES):
            st.error(
                "The uploaded CSV must contain all 15 expected input features."
            )

        else:

            X_test = test_df[EXPECTED_FEATURES]
            y_test = test_df["y"]

            if not set(y_test.unique()).issubset({"no", "yes"}):
                st.error("Target column 'y' must contain only 'no' and 'yes'.")

            else:

                # Load all five saved models

                loaded_models = {}
                comparison_results = []

                for model_name, model_file in MODEL_FILES.items():

                    model_path = MODEL_DIR / model_file

                    if not model_path.exists():
                        st.error(f"Model file not found: {model_path}")
                        st.stop()

                    model = joblib.load(model_path)
                    loaded_models[model_name] = model

                    metrics, _ = evaluate_model(
                        model, X_test, y_test
                    )

                    comparison_results.append({
                        "Model": model_name,
                        **metrics
                    })

                comparison_df = pd.DataFrame(comparison_results)
                comparison_df = comparison_df.set_index("Model")

                # Model comparison section

                st.header("2. Model Comparison")

                st.write(
                    "Performance comparison of all five trained "
                    "classification models on the uploaded test dataset."
                )

                st.dataframe(
                    comparison_df.T.style.format("{:.4f}").highlight_max(
                        axis=1, color="#d4edda"
                    ),
                    use_container_width=True
                )

                # Best model based on F1 Score and MCC

                best_model = comparison_df["F1 Score"].idxmax()

                st.success(
                    f"Best model based on F1 Score: {best_model} "
                    f"(F1 Score: {comparison_df.loc[best_model, 'F1 Score']:.4f})"
                )

                st.caption(
                    "The comparison is calculated dynamically using "
                    "the uploaded test dataset. Your notebook experiment "
                    "selected Gaussian Naive Bayes based on F1 Score and MCC."
                )

                # Individual model selection

                st.header("3. Individual Model Evaluation")

                selected_model = st.selectbox(
                    "Choose a trained model",
                    list(MODEL_FILES.keys())
                )

                model = loaded_models[selected_model]

                metrics, y_pred = evaluate_model(
                    model, X_test, y_test
                )

                st.success(
                    f"Evaluation completed using {selected_model}."
                )

                # Display evaluation metrics

                st.header("4. Model Evaluation Metrics")

                metric_columns = st.columns(3)

                for index, (name, value) in enumerate(metrics.items()):
                    metric_columns[index % 3].metric(
                        name, f"{value:.4f}"
                    )

                # Display confusion matrix

                st.header("5. Confusion Matrix")

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

                st.header("6. Classification Report")

                report = classification_report(
                    y_test,
                    y_pred,
                    labels=["no", "yes"],
                    zero_division=0,
                    output_dict=True
                )

                st.dataframe(
                    pd.DataFrame(report).transpose().round(4),
                    use_container_width=True
                )

    except Exception as error:
        st.error(f"Unable to evaluate the dataset: {error}")

else:
    st.info("Please upload a test CSV file to begin evaluation.")
