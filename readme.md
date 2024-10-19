# SVM with RBF Kernel for Customer Churn Prediction

This project focuses on predicting customer churn using Support Vector Machines (SVM) with an RBF (Radial Basis Function) kernel, followed by model interpretability using SHAP (SHapley Additive exPlanations). The dataset used in this project contains customer information such as age, gender, subscription type, and churn status.

## Project Summary

### 1. Problem Statement
Customer churn refers to customers discontinuing their relationship or subscription with a company. This project aims to predict customer churn using machine learning models, specifically SVM with an RBF kernel, and interpret the model using SHAP to understand the factors contributing to churn.

### 2. Dataset
The dataset consists of customer demographic details and their interactions with the company. The target variable is `Churn`, indicating whether a customer has discontinued using the service.

### 3. Data Preprocessing
- **Handling Missing Values**: Numerical columns were imputed with median values, while categorical columns were imputed with mode values.
- **Feature Encoding**: Categorical variables (`Gender`, `Subscription Type`, `Contract Length`) were encoded using `LabelEncoder`.
- **Feature Scaling**: Numerical features were scaled using `StandardScaler` to ensure they are on the same scale, which is essential for SVM.
- **Data Splitting**: The training data was split into training and validation sets using an 80-20 ratio.

### 4. SVM with RBF Kernel
Support Vector Machines (SVM) is a supervised machine learning algorithm used for classification tasks. In this project, we use an SVM with an RBF kernel to classify customers into churned or non-churned.

#### RBF Kernel
The RBF kernel is a popular choice for SVM as it can handle non-linear relationships by mapping the input features into a higher-dimensional space. The RBF kernel is defined as follows:

\[
K(x, x') = \exp\left(-\gamma \cdot \lVert x - x' \rVert^2\right)
\]

Where:
- \( K(x, x') \) is the kernel function.
- \( \gamma \) is a parameter that defines how far the influence of a single training example reaches.
- \( \lVert x - x' \rVert^2 \) is the squared Euclidean distance between two feature vectors.

The SVM aims to find an optimal hyperplane that separates the data points into different classes while maximizing the margin between them.

### 5. Model Training and Evaluation
- The SVM model with an RBF kernel was trained using the training dataset.
- The model was evaluated using a validation set, and a classification report was generated to assess the model's performance.

The classification report includes metrics like precision, recall, and F1-score, providing insight into the model's performance in predicting churn.

### 6. Model Interpretability with SHAP
To understand the model's predictions, we use SHAP (SHapley Additive exPlanations), a popular tool for model interpretability.

#### SHAP Explanation
SHAP values provide an explanation for each prediction made by the model by assigning an importance value to each feature. The SHAP value represents the contribution of each feature towards the prediction of a particular instance. SHAP values are based on game theory and use the concept of Shapley values to fairly distribute the contribution of each feature.

The SHAP value for a feature \( i \) is defined as:

\[
\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! (|N| - |S| - 1)!}{|N|!} \left( f(S \cup \{i\}) - f(S) \right)
\]

Where:
- \( \phi_i \) is the SHAP value for feature \( i \).
- \( N \) is the set of all features.
- \( S \) is a subset of features excluding feature \( i \).
- \( f(S) \) is the model's prediction using features in subset \( S \).

SHAP provides both global and local interpretability:
- **Global Interpretability**: Understanding the overall impact of each feature on the model.
- **Local Interpretability**: Understanding why a specific prediction was made.

### 7. Visualizing SHAP Results
To effectively understand the SHAP values, various plots are used:
- **Summary Plot**: Displays the overall feature importance and their distribution of impact on the model's output.
- **Force Plot**: Shows the contribution of each feature for a single prediction.
- **Dependence Plot**: Shows how the value of a feature affects the model's output.

#### Placeholder for SHAP Images
- **Summary Plot**: ![Summary Plot](images/summary_plot.png)
- **Force Plot for a Single Prediction**: ![Force Plot](images/force_plot.png)
- **Dependence Plot**: ![Dependence Plot](images/dependence_plot.png)

### 8. Conclusion
- The SVM model with an RBF kernel was successfully used to predict customer churn.
- SHAP was used to interpret the predictions, providing insights into the factors influencing customer churn, which can help businesses make data-driven decisions to retain customers.

## Repository Structure
```
- data/
  - customer_churn_dataset-training-master.csv
  - customer_churn_dataset-testing-master.csv
- images/
  - summary_plot.png
  - force_plot.png
  - dependence_plot.png
- notebooks/
  - kernel-svm.ipynb
- src/
  - preprocess.py
  - train_svm.py
  - interpret_shap.py
- README.md
```

## Getting Started
1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the repository directory:
   ```
   cd <repository-directory>
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the preprocessing script:
   ```
   python src/preprocess.py
   ```
5. Train the SVM model:
   ```
   python src/train_svm.py
   ```
6. Interpret the model using SHAP:
   ```
   python src/interpret_shap.py
   ```

## References
- SHAP GitHub repository: [https://github.com/shap/shap](https://github.com/shap/shap)
- Scikit-learn documentation: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)

### Notes
This project is part of an assignment for an interpretable machine learning course, emphasizing the importance of understanding model behavior and the impact of individual features on predictions.
