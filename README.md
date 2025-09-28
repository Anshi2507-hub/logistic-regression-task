# logistic-regression-task
# 🤖 AI & ML INTERNSHIP - Task 4: Classification with Logistic Regression

## 🎯 Objective
The goal of this task was to **build and evaluate a binary classification model** using the **Logistic Regression** algorithm, leveraging Python libraries like Scikit-learn, Pandas, and Matplotlib.

## 💻 Implementation Summary

The project was executed in a Google Colab notebook (`task-4-logistic-regression.ipynb`) following these steps:

1.  **Dataset Selection:** Used the **Breast Cancer Wisconsin Dataset**, a standard binary classification dataset available in Scikit-learn, where the target variable is **Malignant (0)** or **Benign (1)**.
2.  **Data Preprocessing:** The data was split into training and testing sets (70/30 split), and the features were **Standardized** using `StandardScaler`. This is crucial for Logistic Regression to ensure efficient convergence.
3.  **Model Training:** A `LogisticRegression` model from Scikit-learn was instantiated and fitted to the scaled training data.
4.  **Evaluation:** The model's performance was comprehensively evaluated using:
    * **Confusion Matrix:** To visualize True Positives, True Negatives, False Positives, and False Negatives.
    * **Classification Report:** To calculate **Precision, Recall, and F1-Score** for both classes.
    * **ROC-AUC Score:** Calculated and plotted the Receiver Operating Characteristic (ROC) curve to assess overall model discriminative power across all thresholds (resulting in an AUC score typically close to 1.0).
5.  **Concept Demonstration:** Demonstrated the concept of **threshold tuning** by re-evaluating the model at a threshold of **0.3** to show the resulting trade-off between Precision and Recall.

---

## ❓ Interview Questions and Concepts

The following questions address the core concepts of binary classification and Logistic Regression:

### 1. How does logistic regression differ from linear regression?
| Feature | Linear Regression | Logistic Regression |
| :--- | :--- | :--- |
| **Problem Type** | **Regression** (predicts continuous values, e.g., 10.5) | **Classification** (predicts discrete categories, e.g., 0 or 1) |
| **Output** | A continuous straight line (unbounded $\pm\infty$) | A probability between **0 and 1** (via the Sigmoid function) |

### 2. What is the sigmoid function?
The **Sigmoid Function** (or Logistic Function) is an S-shaped curve that is the core of Logistic Regression. It takes any real number and transforms it into a probability value between **0 and 1**. The formula is: $$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### 3. What is precision vs recall?
* **Precision (Positive Predictive Value):** Measures the accuracy of the positive predictions. **Of all the cases the model predicted as positive, how many were actually positive?** (Focuses on minimizing **False Positives**).
* **Recall (Sensitivity):** Measures the model's ability to find all positive cases. **Of all the cases that were actually positive, how many did the model correctly identify?** (Focuses on minimizing **False Negatives**).

### 4. What is the ROC-AUC curve?
The **ROC (Receiver Operating Characteristic) curve** plots the **True Positive Rate (Recall)** against the **False Positive Rate (1 - Specificity)** at various threshold settings. The **AUC (Area Under the Curve)** is the single score that summarizes the model's performance. An AUC of 1.0 is perfect, and 0.5 is no better than random guessing.

### 5. What is the confusion matrix?
A **Confusion Matrix** is a table that summarizes a classification model's performance by showing the counts of correct and incorrect predictions, broken down by class:
$$
\begin{bmatrix} 
\text{True Negatives (TN)} & \text{False Positives (FP)} \\ 
\text{False Negatives (FN)} & \text{True Positives (TP)} 
\end{bmatrix}
$$

### 6. What happens if classes are imbalanced?
If classes are imbalanced (e.g., 95% Class A, 5% Class B), the model can become biased toward the majority class (A), achieving misleadingly high **Accuracy** (e.g., 95%). To address this:
* Use metrics like **Precision, Recall, and F1-Score** instead of just Accuracy.
* Employ techniques like **Oversampling** the minority class (e.g., SMOTE) or using the `class_weight='balanced'` parameter in the model.

### 7. How do you choose the threshold?
The default threshold is **0.5**. The choice depends entirely on the business or domain objective:
* **To increase Recall (minimize missed cases):** **Lower the threshold** (e.g., to 0.3). This is crucial in medical diagnosis where minimizing False Negatives is the priority.
* **To increase Precision (minimize false alarms):** **Raise the threshold** (e.g., to 0.7). This is useful in spam filtering where minimizing False Positives (marking a good email as spam) is critical.

### 8. Can logistic regression be used for multi-class problems?
**Yes.** Logistic Regression can be extended for multi-class problems (more than two classes) using strategies like:
* **One-vs-Rest (OvR) or One-vs-All (OvA):** Training a separate binary classifier for each class against all other classes combined, and taking the prediction with the highest probability.
* **Multinomial Logistic Regression (Softmax):** A generalized version that directly handles multiple classes simultaneously.

---

## 🔗 Submission Artifacts
* `task-4-logistic-regression.ipynb`: The Google Colab notebook containing all the code, output, visualizations, and detailed explanations.
* `README.md`: This summary document.
