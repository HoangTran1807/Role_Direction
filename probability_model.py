from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd
import re


df = pd.read_csv('./data/mbti_1.csv')


# Create a bag of words vectorizer
def preprocess_data(data):
    # thay thế các ký tự ||| bằng dấu cách
    data = data.replace('|||', ' ||| ')
    # xóa các file .jpg
    data = re.sub(r'\S*\.jpg', ' [jpg] ', data)
    # xóa http link và https link
    data = re.sub(r'http\S+', ' [http] ', data)
    # xóa các link
    data = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' [link] ', data)
    # xóa các khoảng trắng thừa
    data = re.sub(' +', ' ', data)
    return data

df['posts'] = df['posts'].apply(preprocess_data)

#chuyển đổi các chữa hoa thành chữ thường
df['posts'] = df['posts'].apply(lambda x: x.lower())







#vectorizer = CountVectorizer(stop_words='english', max_features=1000)

# # sử dụng TfidfVectorizer thay cho CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)



# # save df['posts'] to a text file
# df['posts'].to_csv('text.txt', index=False)

df['I-E'] = df['type'].apply(lambda x: 0 if x[0] == 'I' else 1)
df['N-S'] = df['type'].apply(lambda x: 0 if x[1] == 'N' else 1)
df['T-F'] = df['type'].apply(lambda x: 0 if x[2] == 'T' else 1)
df['J-P'] = df['type'].apply(lambda x: 0 if x[3] == 'J' else 1)
# xác định 'I-E' là target và 'posts' là feature
X = vectorizer.fit_transform(df['posts'])

#lưu x vào file txt 
with open('X.txt', 'w') as f:
    f.write(str(X))

y = df['type']




# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





# #train a random forest classifier
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42)
# model.fit(X_train, y_train)

# # Make a prediction on your test data
# y_pred = model.predict(X_test)

# # Evaluate the model on your training data
# acc = model.score(X_train, y_train)
# print('Random Forest:', acc)

# # Evaluate the model
# from sklearn.metrics import accuracy_score
# acc = accuracy_score(y_test, y_pred)
# print('Random Forest:', acc)


# #train a neural network classifier
# from sklearn.neural_network import MLPClassifier
# model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
# model.fit(X_train, y_train)
# # Make a prediction on your test data
# y_pred = model.predict(X_test)
# # Evaluate the model on your training data
# acc = model.score(X_train, y_train)
# print('Neural Network:', acc)
# # Evaluate the model
# from sklearn.metrics import accuracy_score
# acc = accuracy_score(y_test, y_pred)
# print('Neural Network:', acc)


#train a support vector machine classifier
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
# Make a prediction on your test data
y_pred = model.predict(X_test)
# Evaluate the model on your training data
acc = model.score(X_train, y_train)
print('SVM:', acc)
# Evaluate the model
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print('SVM:', acc)





# #train a naive bayes classifier
# model = MultinomialNB()
# model.fit(X_train, y_train)
# # Make a prediction on your test data
# y_pred = model.predict(X_test)
# # Evaluate the model on your training data
# acc = model.score(X_train, y_train)
# print('Naive Bayes:', acc)
# # Evaluate the model
# from sklearn.metrics import accuracy_score
# acc = accuracy_score(y_test, y_pred)
# print('Naive Bayes:', acc)






# import numpy as np
# from sklearn.metrics import accuracy_score
# from sklearn.ensemble import StackingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC


# # Define the base models
# level0 = list()
# level0.append(('lr', LogisticRegression(max_iter=10000)))
# level0.append(('dt', DecisionTreeClassifier()))
# level0.append(('svm', SVC()))

# # Define meta learner model
# level1 = LogisticRegression()

# # Define the stacking ensemble
# model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

# # Fit the model on your training data
# model.fit(X_train, y_train)

# # Make a prediction on your test data
# y_pred = model.predict(X_test)

# # Evaluate the model on your test data
# acc = accuracy_score(y_test, y_pred)

# print(acc)