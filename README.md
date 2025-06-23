
# Debutanizer Data Modeling Project

This is my ongoing project which focuses on regression and classification tasks using real-world data from a debutanizer process. It includes outlier detection, regression modeling (Decision Tree, Random Forest, XGBoost), and classification using Naive Bayes after discretizing the target variable.

---

## Dataset

The dataset is a whitespace-separated `.txt` file containing 8 columns:

- **Inputs (u1 to u7):** Continuous process variables
- **Target (y):** Output variable (continuous)

Example rows:
```
u1    u2    u3    u4    u5    u6    u7    y
0.26  0.65  0.83  0.58  0.78  0.84  0.82  0.18
...
```

---

## Steps Followed

### 1. **Loading and Cleaning Data**

- File read using regex to handle whitespace separation.
- Z-score used to identify and remove outliers (`z > 3 or z < -3`).
- Final cleaned dataset has 2335 rows.

```python
df = pd.read_csv('debutanizer_data.txt', sep='\s+', header=None)
df.columns = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'y']
df['zscore'] = stats.zscore(df['y'])
df_clean = df[(df['zscore'] <= 3) & (df['zscore'] >= -3)]
```

---

### 2. **Regression Models**

#### ✅ Decision Tree Regressor

- Tuned with GridSearchCV
- Best Parameters: `max_depth=10, min_samples_leaf=4, min_samples_split=2, max_features=None`
- R² Score: `0.489`
- MAE: `0.0576`
- MSE: `0.00875`

#### ✅ Random Forest Regressor

- Same parameter tuning
- R² Score: `0.489`
- MAE: `0.0576`
- MSE: `0.00875`

#### ✅ XGBoost Regressor

- Best Parameters: `colsample_bytree=0.8, learning_rate=0.1, max_depth=3, n_estimators=100, subsample=0.8`
- R² Score: `0.579`
- MAE: `0.0588`
- MSE: `0.00721`

---

### 3. **Classification Using Naive Bayes**

- `y` column binned into 3 classes using `KBinsDiscretizer`
- Naive Bayes Classifier used
- Accuracy: `0.539`
- Precision: `0.553`
- Recall: `0.539`
- F1 Score: `0.523`

---

## 📊 Evaluation Metrics

- **R² Score**: Explained variance
- **MAE/MSE**: Error metrics for regression
- **Accuracy/Precision/Recall/F1**: Classification performance

---

## 🔧 Dependencies

```bash
pip install pandas numpy scipy scikit-learn xgboost statsmodels
```

---

## 👨‍💻 Author

Ankur Prasad  
IIT Ropar  
Debutanizer Process Modeling – Summer 2025
