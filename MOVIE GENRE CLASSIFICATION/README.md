**Project Overview**

This project focuses on classifying movies into different genres using machine learning techniques. The objective is to build a predictive model that can accurately categorize movies based on their metadata, such as plot descriptions, director, and cast information. This model can be valuable for movie recommendation systems and content organization.

**Dataset**

The dataset used for this project contains movies with associated genres. It includes the following columns:

- `title`: The title of the movie.
- `genre`: The genre(s) of the movie.
- `description`: A brief plot description of the movie.
- `director`: The director of the movie.
- `cast`: The main cast of the movie.
  
**Project Structure**

**Data Preprocessing:**

- Cleaned and preprocessed the text data (e.g., descriptions).
- Tokenized and normalized text data.
- Converted text data into numerical format using techniques like TF-IDF (Term Frequency-Inverse Document Frequency) and word embeddings.
  
**Data Exploration and Visualization:**

- Analyzed the distribution of different movie genres.
- Visualized word frequency across different genres to identify unique patterns and commonalities.
  
**Model Training:**

- Split the data into training (80%) and testing (20%) sets.
- Trained several models, including Naive Bayes, Logistic Regression, and Random Forest, on the training data.
- Evaluated models using metrics such as accuracy, precision, recall, F1-score, and confusion matrix.
  
**Model Selection:**

- Compared different models based on their performance metrics.
- Selected the best-performing model for genre classification, optimizing for accuracy and generalization ability.
  
**Dependencies**

The project requires the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `nltk` (Natural Language Toolkit)

You can install these dependencies using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk
```



**Conclusion**

This project successfully demonstrates movie genre classification using machine learning. Through meticulous data preprocessing, model training, and evaluation, a robust classification system was developed. This model can be integrated into recommendation systems, enhancing user experience by providing accurate genre-based suggestions.

**Author**
Vikram Sen
