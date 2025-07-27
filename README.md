User Behaviour Detection using Machine Learning
This project develops a machine learning pipeline to classify user behavior as "active" or "inactive" based on their in-app interactions. By analyzing patterns in session duration, clicks, and device information, we can predict user engagement, which is crucial for personalization, user retention strategies, and business decision-making.

The project compares four different classification algorithms—Random Forest, Logistic Regression, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN)—to identify the most effective model for this task.

🎯 Project Goal
The primary objective is to build and evaluate a machine learning model that accurately predicts user behavior (active/inactive) from a real-world dataset. The project covers an end-to-end ML workflow, including data preprocessing, handling class imbalance, model training, and performance evaluation.

🛠️ Tech Stack & Libraries
Core Language: Python

Data Manipulation: Pandas, NumPy

Machine Learning: Scikit-learn

Imbalance Handling: SMOTE (from imblearn)

Data Visualization: Matplotlib, Seaborn

📈 Methodology & Pipeline
The project followed a structured machine learning pipeline:

Data Loading & Cleaning: The user_behavior_dataset.csv was loaded, and the non-predictive User ID column was removed.

Data Preprocessing:

Label Encoding: Categorical features (Device Model, Operating System, Gender) were converted into numerical format.

Feature Scaling: Numerical features (Session Duration, Clicks) were normalized using StandardScaler to ensure they are on a comparable scale.

Train-Test Split: The dataset was divided into an 80% training set and a 20% testing set.

Handling Class Imbalance: The SMOTE (Synthetic Minority Over-sampling Technique) was applied to the training data to correct the imbalance between active and inactive user samples, preventing model bias.

Model Training: Four different models were trained on the balanced dataset:

Random Forest

Logistic Regression

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Model Evaluation: Each model was evaluated on the unseen test set using Accuracy, Recall, F1-Score, and 5-fold Cross-Validation Accuracy.

📊 Results & Model Comparison
The Random Forest classifier achieved the best overall performance across all metrics, making it the most reliable model for this task.

Model

Accuracy

Recall

F1-Score

Cross-Validation Accuracy

Random Forest

0.87

0.86

0.85

0.84

Support Vector Machine (SVM)

0.84

0.83

0.83

0.82

Logistic Regression

0.82

0.80

0.81

0.80

K-Nearest Neighbors (KNN)

0.78

0.77

0.76

0.75

Key Insights
Feature Importance Analysis revealed that Session Duration and Clicks are the most significant predictors of user activity. This aligns with the logical assumption that more engaged users spend more time and interact more with the app.

🏁 How to Run
Prerequisites
Python 3.8+

Jupyter Notebook or any Python IDE

Installation & Execution
Clone the repository:

git clone https://github.com/upadhyay-jash/user-behaviour-detection.git
cd user-behaviour-detection

Install dependencies:

pip install -r requirements.txt

Run the analysis:

Open the Jupyter Notebook containing the project code.

Run the cells sequentially to perform data preprocessing, model training, and evaluation.
