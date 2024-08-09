Project Overview
This project aims to predict customer churn in a financial services company using machine learning techniques. Customer churn refers to the loss of clients or customers. The goal is to build a predictive model that can accurately identify customers who are likely to churn, enabling the company to take proactive measures to retain them.

Dataset
The dataset used in this project is sourced from [Churn_Modelling.csv](https://www.kaggle.com/datasets/kartik2112/fraud-detection). It contains the following features:

CreditScore: Credit score of the customer.
Geography: The country from which the customer belongs.
Gender: Gender of the customer.
Age: Age of the customer.
Tenure: Number of years the customer has been with the company.
Balance: Account balance of the customer.
NumOfProducts: Number of products the customer has with the company.
HasCrCard: Whether the customer has a credit card.
IsActiveMember: Whether the customer is an active member.
EstimatedSalary: Estimated salary of the customer.
Exited: The target variable, indicating whether the customer has churned (1) or not (0).
Project Structure
Data Preprocessing:

Removed irrelevant columns like Surname and RowNumber.
Converted categorical variables (Geography, Gender) into numerical format using one-hot encoding.
Checked for and handled missing values.
Data Exploration and Visualization:

Visualized the distribution of numerical features such as CreditScore, Age, Balance, and EstimatedSalary.
Examined the relationship between these features and the target variable Exited using box plots.
Analyzed the correlation between features using a correlation matrix heatmap.
Model Training:

Split the data into training (80%) and testing (20%) sets.
Trained a Logistic Regression model on the training data.
Evaluated the model using accuracy, classification report, and confusion matrix.
Hyperparameter Tuning:

Used GridSearchCV to tune hyperparameters (C, penalty, and solver) of the Logistic Regression model.
Trained the model with the best parameters and evaluated its performance.
Dependencies
The project requires the following Python libraries:

numpy
pandas
matplotlib
seaborn
scikit-learn
You can install these dependencies using pip:

bash
Copy code
pip install numpy pandas matplotlib seaborn scikit-learn
Results
Initial Model Accuracy: The logistic regression model achieved an accuracy of X.XXX on the test data.
Best Model Accuracy: After hyperparameter tuning, the best model achieved an accuracy of X.XXX.
The final model's confusion matrix and classification report provide detailed insights into its performance.
Conclusion
The project successfully demonstrates the process of predicting customer churn using a logistic regression model. The use of data preprocessing, exploratory data analysis, and hyperparameter tuning contributed to building a more accurate model. This predictive model can be used by the company to identify at-risk customers and take necessary actions to reduce churn.

Author
Vikram Sen

