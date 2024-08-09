Project Overview
This project aims to detect spam SMS messages using machine learning techniques. The goal is to build a predictive model that can accurately classify SMS messages as spam or legitimate, helping to filter out unwanted communications and protect users from potential scams.

Dataset
The dataset used in this project contains SMS messages labeled as either ham (legitimate) or spam. It consists of the following columns:

v1: Label indicating whether the message is 'ham' (legitimate) or 'spam'.
v2: The raw text of the SMS message.
Project Structure
Data Preprocessing:

Cleaned and normalized the text data.
Converted text data into numerical format using techniques like TF-IDF (Term Frequency-Inverse Document Frequency).
Data Exploration and Visualization:

Analyzed the distribution of spam and ham messages.
Visualized word frequency for both spam and ham messages to identify common patterns.
Model Training:

Split the data into training (80%) and testing (20%) sets.
Trained several models, including Naive Bayes and Logistic Regression, on the training data.
Evaluated the models using accuracy, precision, recall, F1-score, and confusion matrix.
Model Selection:

Compared different models and selected the one with the best performance metrics.
Dependencies
The project requires the following Python libraries:

numpy
pandas
matplotlib
seaborn
scikit-learn
nltk (Natural Language Toolkit)
You can install these dependencies using pip:

bash
Copy code
pip install numpy pandas matplotlib seaborn scikit-learn nltk

Conclusion
This project successfully demonstrates the detection of spam SMS messages using machine learning. Through careful data preprocessing, model training, and evaluation, a robust spam detection system was developed. This model can be effectively used to filter out spam messages and enhance user security.

Author
Vikram Sen
