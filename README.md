# Bank Marketing Term Deposit Prediction

## 1. Problem Statement

In this assignment, I have worked on a classification problem using the Bank Marketing dataset.

The main objective of this project is to predict whether a customer will subscribe to a term deposit or not, based on the customer's details and previous marketing campaign information.

I have implemented the following five machine learning algorithms:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbours (KNN)
4. Gaussian Naive Bayes
5. Linear Support Vector Machine (SVM)

I have evaluated all five models using the same training and testing data and compared their performance using six evaluation metrics.

## 2. Dataset Description

For this assignment, I have selected the Bank Marketing dataset from the UCI Machine Learning Repository.

**Dataset Link:** https://archive.ics.uci.edu/dataset/222/bank+marketing

The dataset contains information about customers contacted during a bank marketing campaign. The target column is `y`, which indicates whether the customer subscribed to a term deposit (`yes` or `no`).

The dataset details are given below:

- Total number of records: 45,211
- Total original columns: 17
- Target column: `y`
- Input features used: 15
- Training records: 36,168
- Testing records: 9,043
- Train-test split: 80:20

I observed that the dataset did not contain any missing values or duplicate records.

The target variable was imbalanced:
- No: 39,922 records (88.3%)
- Yes: 5,289 records (11.7%)

I removed the `duration` column because the call duration is known only after the customer call. I wanted to use the remaining information to predict the subscription.

For preprocessing, I used StandardScaler for numerical features and OneHotEncoder for categorical features. I also used stratified train-test splitting to maintain the class distribution in both datasets.

## 3. GitHub Repository

I have maintained the project files in a GitHub repository.

**GitHub Repository Link:** To be added after uploading the project.

The repository includes my Jupyter Notebook, Streamlit application, saved models, test dataset and requirements file.

## 4. Comparison of Classification Models

After training all five models, I evaluated them using the test dataset.

The results obtained are shown below:

| Evaluation Metric | Logistic Regression | Decision Tree | KNN | Gaussian Naive Bayes | Linear SVM |
|---|---:|---:|---:|---:|---:|
| Accuracy | 0.7548 | 0.8401 | 0.8874 | 0.8452 | 0.7578 |
| AUC | 0.7722 | 0.6036 | 0.7027 | 0.7514 | 0.7718 |
| Precision | 0.2662 | 0.3083 | 0.5529 | 0.3708 | 0.2684 |
| Recall | 0.6238 | 0.2949 | 0.1975 | 0.4641 | 0.6200 |
| F1 Score | 0.3732 | 0.3014 | 0.2911 | 0.4123 | 0.3746 |
| MCC | 0.2853 | 0.2113 | 0.2833 | 0.3271 | 0.2867 |

## 5. Observations from the Models

### 5.1 Logistic Regression

Logistic Regression achieved an accuracy of 0.7548 and AUC of 0.7722.

I observed that this model achieved the highest recall of 0.6238 among the five models. However, its precision was 0.2662, which is comparatively low.

### 5.2 Decision Tree

Decision Tree achieved an accuracy of 0.8401.

However, its AUC was 0.6036 and recall was 0.2949. Its F1 score and MCC were also lower compared with the other models.

### 5.3 K-Nearest Neighbours

KNN achieved the highest accuracy of 0.8874 and precision of 0.5529.

However, its recall was only 0.1975. This means that the model could not identify many customers who actually subscribed to a term deposit.

### 5.4 Gaussian Naive Bayes

Gaussian Naive Bayes achieved an accuracy of 0.8452 and AUC of 0.7514.

I observed that this model achieved the highest F1 score of 0.4123 and MCC of 0.3271 among all five models.

It provided a better balance between precision and recall compared with the other models in this experiment.

### 5.5 Linear Support Vector Machine

Linear SVM achieved an accuracy of 0.7578 and AUC of 0.7718.

Its recall was 0.6200, which was close to Logistic Regression. The F1 score was 0.3746 and MCC was 0.2867.

## 6. Overall Best Model

After comparing the results of all five models, I found that Gaussian Naive Bayes performed better in terms of F1 score and MCC.

KNN achieved the highest accuracy, but its recall was low. Since the dataset is imbalanced, I did not consider accuracy alone while selecting the best model.

Based on the F1 score and MCC, I selected Gaussian Naive Bayes as the best overall model for this experiment.

## 7. Streamlit Application

I have developed a Streamlit application using Python.

The application provides the following features:

1. Upload a test dataset in CSV format.
2. Select any one of the five trained models.
3. Load the saved model and evaluate the uploaded dataset without retraining.
4. Display Accuracy, AUC, Precision, Recall, F1 Score and MCC.
5. Display the confusion matrix.
6. Display the classification report.

**Streamlit Application Link:** To be added after deployment.

## 8. Project Files

The project contains the following files and folders:

- `Prediction.ipynb` – Contains dataset loading, data exploration, preprocessing, model training and evaluation.
- `app.py` – Contains the Streamlit application code.
- `requirements.txt` – Contains the required Python libraries.
- `test_data.csv` – Contains the test dataset used in the Streamlit application.
- `model/` – Contains all five saved trained models.
- `README.md` – Contains the project description and model comparison results.

## 9. Conclusion

Through this assignment, I got an opportunity to implement and compare five different classification algorithms using the Bank Marketing dataset.

I performed data exploration, preprocessing, model training and evaluation. I also saved all five trained models and developed a Streamlit application to evaluate them using uploaded test data.

From the results, I found that Gaussian Naive Bayes achieved the highest F1 score and MCC. Therefore, I selected it as the best overall model among the five models used in this assignment.