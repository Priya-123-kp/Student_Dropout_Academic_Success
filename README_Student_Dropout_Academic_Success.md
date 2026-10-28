# Student Dropout and Academic Success Prediction Using Deep Learning

**Student:** Priya K  
**Batch:** DSB02  
**Project Category:** Deep Learning  
**Primary Technique:** Artificial Neural Network (ANN)  
**Prediction Classes:** Dropout, Enrolled, Graduate

## 1. Project Overview

Higher education institutions may find it difficult to identify students who are at risk of dropping out before they leave their studies. This project uses historical student information to build a multiclass classification solution that predicts a student's academic outcome as **Dropout**, **Enrolled**, or **Graduate**.

The model uses demographic, academic, and socioeconomic information. The intended use is to help institutions identify students who may need additional academic guidance or support. Predictions should support—not replace—professional judgement and direct student support.

## 2. Project Objectives

- Understand the student dropout and academic success prediction problem.
- Explore and prepare the student dataset.
- Examine relationships between student characteristics and academic outcomes.
- Build and compare machine-learning and deep-learning classification models.
- Evaluate model performance on held-out data.
- Prepare the selected model and preprocessing objects for API deployment.

## 3. Intended Users

- Educational institutions and college administrators
- Teachers and academic advisors
- Student support and intervention teams

## 4. Dataset

**Source:** UCI Machine Learning Repository  
**Dataset:** [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

The project notebook describes 4,424 student records, 36 input features, and one target column (`Target`). The target contains three outcomes:

| Outcome | Meaning |
|---|---|
| `Dropout` | The student dropped out |
| `Enrolled` | The student remains enrolled |
| `Graduate` | The student graduated |

The features cover student demographic, academic, admission, and socioeconomic information. The dataset contains both numerical features and categorical features represented by coded values.

### Target distribution

The notebook records the following class counts:

| Target class | Number of records |
|---|---:|
| Graduate | 2,209 |
| Dropout | 1,421 |
| Enrolled | 794 |

The classes are imbalanced, particularly the Enrolled class. Accuracy should therefore be considered alongside class-level precision, recall, F1-score, and the confusion matrix.

## 5. Workflow

1. **Problem definition** — define the academic outcome to predict and identify intended users.
2. **Dataset loading** — load the student dataset and inspect its structure.
3. **Exploratory Data Analysis (EDA)** — review data types, descriptive statistics, target distribution, and relationships between features.
4. **Data cleaning** — check duplicate records and missing values.
5. **Preprocessing** — separate the target from the input features, encode categorical variables, and scale numerical variables.
6. **Data splitting** — create stratified training, validation, and test sets.
7. **Model building** — train candidate machine-learning models and ANN variants.
8. **Model selection** — compare candidates using validation accuracy.
9. **Final evaluation** — evaluate the selected model on the held-out test set.
10. **Deployment preparation** — save the trained model and the preprocessing/label-mapping objects needed to reproduce predictions.
11. **API testing** — send valid input data to the prediction endpoint and check the returned class.

## 6. Exploratory Data Analysis

The notebook includes analysis of:

- Dataset shape, data types, and descriptive statistics
- Missing values and duplicate records
- Distribution of the target classes
- Age at enrollment and admission grade
- Scholarship status and tuition-fee payment status
- Approved curricular units in the first and second semesters
- Correlations between numerical features

The notebook observations suggest that academic performance indicators, including approved curricular units, may be associated with student outcomes. These are observed relationships, not proof that any single feature causes a particular outcome.

## 7. Data Preprocessing

The notebook uses a stratified split with a fixed random seed:

- 70% training data
- 15% validation data
- 15% test data

The target is encoded into integer class labels using `LabelEncoder`. Input features are processed with a `ColumnTransformer`:

- **Categorical features:** `OneHotEncoder(handle_unknown="ignore")`
- **Numerical features:** median imputation and `StandardScaler`

Some categorical attributes are stored as integers in the source dataset, so the notebook specifies a meaning-based list of categorical columns instead of treating every numeric column as continuous.

**Important:** The preprocessing transformer is fitted on the training data only. The validation and test data are transformed using that fitted transformer to avoid data leakage.

## 8. Models

The notebook compares several candidate models:

- Logistic Regression
- Random Forest
- XGBoost
- Baseline ANN
- Improved ANN
- Fine-tuned ANN
- Optimized ANN

The ANN is a feed-forward neural network for three-class classification. The notebook explores multiple hidden-layer configurations, dropout, early stopping, L2 regularization, and learning-rate reduction. The model comparison code selects the candidate using validation accuracy, then evaluates the selected candidate on the held-out test set.

### Evaluation metrics

- **Accuracy:** proportion of all predictions that are correct.
- **Precision:** proportion of predicted members of a class that truly belong to that class.
- **Recall:** proportion of actual members of a class correctly identified.
- **F1-score:** balance between precision and recall.
- **Confusion matrix:** shows correct and incorrect predictions for each class.

**Final results:** Record the selected model name, validation accuracy, test accuracy, and per-class metrics from the latest complete notebook run here. Do not use an earlier run's accuracy as the final result if the model or data split has changed.

| Result | Value from final run |
|---|---|
| Selected model | To be filled from the latest run |
| Validation accuracy | To be filled from the latest run |
| Test accuracy | To be filled from the latest run |
| Dropout precision / recall / F1 | To be filled from the final classification report |
| Enrolled precision / recall / F1 | To be filled from the final classification report |
| Graduate precision / recall / F1 | To be filled from the final classification report |

## 9. Deployment

The project notebook identifies **FastAPI** as the API technology. The prediction service should load the exact selected model and the matching preprocessing and label-encoding artifacts saved during training.

Typical artifacts used by the deployment workflow are:

```text
artifacts/
├── preprocessor.pkl
├── label_encoder.pkl
├── feature_columns.pkl
└── student_model.keras   # for a selected Keras ANN
```

If the selected model is a scikit-learn pipeline instead of a Keras ANN, save and load that fitted pipeline using the matching artifact format. The API must use the same feature names, feature order, preprocessing steps, and target-label mapping as the training notebook.

### Suggested project structure

```text
Student-Dropout-Academic-Success/
├── app.py
├── requirements.txt
├── README.md
└── artifacts/
    ├── preprocessor.pkl
    ├── label_encoder.pkl
    ├── feature_columns.pkl
    └── student_model.keras
```

The actual files in `artifacts/` depend on which model was selected and exported. Do not include generated model files in the repository unless appropriate for the repository's size and sharing requirements.

### Install and run

Create and activate a Python virtual environment, then install the dependencies required by the final `app.py`:

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install packages:

```bash
pip install fastapi "uvicorn[standard]" pandas numpy scikit-learn joblib tensorflow
```

Run the API from the folder containing `app.py`:

```bash
uvicorn app:app --reload
```

Open the interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

The API's `/predict` request schema must match the final `app.py`. Send every required feature with its exact expected name and a valid value. A `422 Unprocessable Entity` response commonly means the submitted JSON does not match the request schema—for example, required fields are missing or a value has the wrong type. Use `/docs` to inspect the actual request schema before testing.

**Implementation note:** The FastAPI cell in the supplied notebook is a minimal example that accepts only `age` and `previous_grade` and returns the submitted values. It is not, by itself, connected to the trained project model. For real predictions, replace that demonstration logic with the model-loading and preprocessing code that matches the exported artifacts and the full feature schema.

## 10. How a User Would Use the Prediction API

1. Start the FastAPI server.
2. Open `/docs` in a browser.
3. Expand `POST /predict`.
4. Select **Try it out**.
5. Enter all required student input features in the exact JSON format shown by the endpoint schema.
6. Select **Execute**.
7. Review the predicted class returned by the API.

The API should return one of the three academic outcome labels: `Dropout`, `Enrolled`, or `Graduate`. The actual request fields depend on the final API schema; do not submit only a few fields if the endpoint requires the complete feature set.

## 11. Limitations and Responsible Use

- The model learns patterns from the dataset and may not generalize to every institution or student population.
- The target classes are imbalanced, so performance should be reviewed for each class—not only by overall accuracy.
- Predictions are estimates and may be incorrect.
- The system should be used to offer timely support, not to penalize, label, or deny opportunities to students.
- Institutions should review data quality, privacy, fairness, and performance on their own student population before operational use.

## 12. Conclusion

This project explores how student demographic, academic, and socioeconomic data can be used to predict Dropout, Enrolled, and Graduate outcomes. The workflow covers data exploration, preprocessing, model comparison, evaluation, and FastAPI deployment preparation. With appropriate validation and responsible use, such a system may help educational institutions identify students who could benefit from additional guidance and support.

## 13. Technologies

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- TensorFlow / Keras
- XGBoost
- FastAPI
- Uvicorn
- Jupyter Notebook / Google Colab

## 14. Project Author

**Priya K**  
**Batch:** DSB02  
**Project:** Student Dropout and Academic Success Prediction Using Deep Learning
