# Titanic Dataset Preprocessing

This project demonstrates a complete preprocessing pipeline for the Titanic dataset using Python and key data science libraries. The goal is to clean and prepare the dataset for machine learning models.

---

## 📁 Dataset Features Used

```plaintext
['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
```

---

## ⚙️ Steps Performed

### 1. **Import Libraries and Load Dataset**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
```

### 2. **Exploratory Data Analysis**

* Used `.info()`, `.describe()`, `.isnull().sum()` to understand the dataset.

### 3. **Handle Missing Values**

* `Age`: filled with **median**
* `Embarked`: filled with **mode**
* Dropped high-missing and less informative columns: `Cabin`, `PassengerId`, `Name`, `Ticket`

### 4. **Encoding Categorical Variables**

* `Sex`: Label encoded (male → 0, female → 1)
* `Embarked`: One-hot encoded, dropping the first to avoid dummy trap

### 5. **Normalize Numerical Features**

* Scaled `Age`, `Fare`, `SibSp`, and `Parch` using `MinMaxScaler`

### 6. **Outlier Detection & Removal**

* Used **boxplot** to visualize outliers in `Fare`
* Removed outliers using the **IQR (Interquartile Range)** method

### 7. **Visualization**

* Boxplots before and after outlier removal for `Fare`

---

## 📊 Libraries Used

* `pandas`, `numpy`
* `matplotlib`, `seaborn`
* `scikit-learn`

---

## ✅ Output

A clean and ready-to-model dataset with no missing values, encoded categorical variables, scaled numerical features, and outliers handled.

---

## 📌 Next Steps (Optional)

* Train models like Logistic Regression, Random Forest, etc.
* Evaluate with accuracy, precision, recall, F1-score
* Perform hyperparameter tuning

---


## 👨‍💻 Author

Jenish Allen Immanuel J

---

> 🚀 Ready for machine learning!
