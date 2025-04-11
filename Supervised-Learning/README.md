# Supervised Learning 📘

Supervised Learning is one of the most fundamental types of Machine Learning. In this paradigm, the model is trained on a **labeled dataset**, which means every input in the dataset has a corresponding correct output (label).

The goal of supervised learning is to **learn a mapping function** from inputs to outputs, so that the model can make accurate predictions on new, unseen data.

---

## 🔑 Key Concepts

- **Input (X)**: Features or independent variables used to make predictions.
- **Output (y)**: Target or dependent variable we are trying to predict.
- **Training Data**: Labeled data used to train the model.
- **Testing/Validation Data**: Used to evaluate the performance of the model.
- **Loss Function**: Measures the difference between predicted and actual values.

---

## 📂 Types of Supervised Learning

### 1. Regression 📈
Regression is used when the target variable is **continuous** (e.g., price, temperature, salary).

#### ✅ Algorithms Covered:

- **Linear Regression**: Models a linear relationship between inputs and target.
- **Polynomial Regression**: Models a non-linear relationship using polynomial terms.
- **Ridge & Lasso Regression**: Regularized versions of linear regression to reduce overfitting.
- **Support Vector Regression (SVR)**: Uses support vectors to predict continuous values.

#### 📌 Example Use Cases:
- Predicting house prices
- Forecasting stock prices
- Estimating temperature

---

### 2. Classification 🧠
Classification is used when the target variable is **categorical** (e.g., spam/ham, yes/no, digit class).

#### ✅ Algorithms Covered:

- **Logistic Regression**: A probabilistic model for binary classification.
- **K-Nearest Neighbors (KNN)**: Classifies based on the labels of the nearest data points.
- **Decision Tree Classifier**: Uses a tree-like structure to make decisions.
- **Random Forest Classifier**: An ensemble of decision trees for better performance.
- **Support Vector Machine (SVM)**: Maximizes the margin between classes.
- **Naive Bayes**: A probabilistic classifier based on Bayes’ theorem.

#### 📌 Example Use Cases:
- Email spam detection
- Disease diagnosis (positive/negative)
- Handwritten digit recognition (MNIST)

---

## 📌 Folder Structure

```
Supervised-Learning/ ├── Regression/ │ ├── LinearRegression.ipynb │ ├── PolynomialRegression.ipynb │ ├── RidgeRegression.ipynb │ └── SVR.ipynb ├── Classification/ │ ├── LogisticRegression.ipynb │ ├── KNN.ipynb │ ├── DecisionTree.ipynb │ ├── RandomForest.ipynb │ ├── SVM.ipynb │ └── NaiveBayes.ipynb └── README.md

```

---

## 🧪 What's Inside Each Notebook?

- Explanation of the algorithm
- Step-by-step implementation using Jupyter Notebook
- Data loading and preprocessing
- Model training and evaluation (accuracy, confusion matrix, etc.)
- Visualization of results where applicable

---

## 🛠️ Tools & Libraries Used

- **NumPy / Pandas** – Data manipulation
- **Matplotlib / Seaborn** – Data visualization
- **Scikit-learn** – ML models and metrics
- **Plotly** – Interactive visualizations (optional)

Install dependencies with:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
