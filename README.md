# MACHINE-LEARNING-MODEL-IMPLEMENTATION

COMPANY : CODTECH IT SOLUTIONS

NAME : SAMARTH KUMAR SINGH

INTERN ID : CT04XBC

DOMAIN : PYTHON PROGRAMMING

DURATION : 4 WEEKS

MENTOR : NEELA SANTOSH

This program:
a) Creates a simple dataset of emails and their labels (spam/not spam)
b) Converts text to numerical features using CountVectorizer
c) Splits data into training and testing sets
d) Trains a Multinomial Naive Bayes classifier
e) Evaluates the model's performance
f) Makes predictions on new emails

a) CountVectorizer: Converts text to numerical features
b) MultinomialNB: Naive Bayes classifier for multinomial distributions
c) train_test_split: Splits data into training and testing sets
d) accuracy_score & classification_report: Evaluation metrics
e) numpy: Numerical computing library

This Python program implements a spam email detection system using scikit-learn's machine learning capabilities. It begins by importing necessary libraries and defining a small dataset of emails labeled as spam (1) or not spam (0). The text data is transformed into numerical features using CountVectorizer, which creates a word count matrix, and then split into training and testing sets. A Multinomial Naive Bayes classifier is trained on the training data to learn patterns distinguishing spam from legitimate emails. The model's performance is evaluated using accuracy and a detailed classification report, comparing predictions against the test set labels. Finally, the script demonstrates real-world application by classifying two new emails and providing both their predicted labels and spam probabilities, showcasing a complete pipeline from data preprocessing to prediction in a simple yet effective spam detection model.
The program relies on a set of powerful Python libraries from the scikit-learn ecosystem and NumPy, each serving a distinct purpose in the spam email detection model. Scikit-learn, installed via pip install scikit-learn, provides the core machine learning functionality: CountVectorizer converts raw text into a numerical matrix of word counts, MultinomialNB implements the Naive Bayes classifier suited for text classification, train_test_split divides the dataset into training and testing subsets for model validation, and accuracy_score along with classification_report offer tools to evaluate the model's performance with metrics like precision and recall. Additionally, NumPy, installed with pip install numpy, is a fundamental library for numerical computing, enabling efficient array operations and mathematical computations that underpin scikit-learn's data processing and model training. Together, these libraries form a robust foundation for building, training, and assessing the spam detection system, leveraging both machine learning algorithms and numerical efficiency.

This spam email detection code offers several advantages, making it a valuable tool for both learning and practical application. By leveraging scikit-learn's robust machine learning libraries, it provides an efficient and straightforward way to classify emails, achieving reasonable accuracy with minimal data and computational resources, which is ideal for small-scale projects or educational purposes. The use of the Naive Bayes classifier ensures fast training and prediction times, even with larger datasets, due to its simplicity and assumption of feature independence, while the CountVectorizer enables seamless conversion of text to numerical features, simplifying the preprocessing step. The code's modular structure allows for easy modification—such as swapping algorithms or adding preprocessing steps—making it adaptable to different datasets or requirements. Additionally, its built-in evaluation metrics and probability outputs offer clear insights into performance and confidence levels, empowering users to assess and refine the model effectively. Overall, this implementation balances simplicity, speed, and flexibility, providing a practical foundation for spam detection that can be scaled or enhanced as needed.

OUTPUT :

![Image](https://github.com/user-attachments/assets/a2fffb3f-2de8-424b-86f5-d61e2a1b28d6)





