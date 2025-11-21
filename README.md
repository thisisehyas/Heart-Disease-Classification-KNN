# ❤️ **Heart Disease Classification using KNN**

The project uses the **K-Nearest Neighbors (KNN)** algorithm on medical data from the **UCI Heart Disease dataset** for the prediction of heart disease.

The aim is to develop a clean and reproducible machine learning pipeline that involves preprocessing, feature selection, dimensionality analysis (PCA), training, and evaluation.

## **Overview of the Project**

Heart disease detection is a high-priority task in medical data science. In the following project, a binary classification model was developed that predicts whether a person has heart disease according to 13 medical attributes comprising age, chest pain type, cholesterol level, blood pressure, and many more.

The workflow involves the following:

- Loading and preparing the UCI dataset
- Handling missing values
- Scaling features
- Selecting important features based on Spearman correlation
- Applying PCA for visualization
- Training various KNN models with different hyperparameters
- Evaluating model performance using standard metrics

The final KNN model achieved **88–89% accuracy** while balanced in performance for both classes.

## **Dataset**

**Source:** UCI Machine Learning Repository  
**Samples:** 303  
**Features:** 13 medical attributes  
**Original Target:** `num` (values 0–4 indicate disease severity)

The target was converted into a **binary classification**:

- `0` → No disease
- `1–4` → Disease

Data is loaded automatically by using:

```python
from ucimlrepo import fetch_ucirepo
heart_disease = fetch_ucirepo(id=45)
```

## **Data Preprocessing**

### Handling missing values

- Columns **ca** and **thal** contained missing values.
- Missing values were replaced using the **median** of each column.

### Feature scaling

KNN uses distance calculations, so features need to be on the same scale.  
**StandardScaler** was applied to normalize all selected features.

## **Feature Analysis & Selection**

I have used **Spearman correlation** to assess the relationship of each feature with the binary target.

Top correlated features (positive and negative):

- **thal, ca, cp, exang, oldpeak, slope** (strong positive correlation)
- **thalach** (strong negative correlation)
- All the low-impact features like **fbs** and **chol** were removed.

The final selected features were:

```
['thal', 'ca', 'cp', 'exang', 'oldpeak', 'slope', 'sex', 'age', 'thalach']
```
