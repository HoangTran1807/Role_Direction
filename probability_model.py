from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd
import re
import time

df = pd.read_csv('./data/mbti_1.csv')



def preprocess_data(data):
    # thay thế các ký tự ||| bằng dấu cách
    data = data.replace('|||', ' ||| ')
    # xóa các file .jpg
    data = re.sub(r'\S*\.jpg', ' image ', data)
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







# vectorizer = CountVectorizer(stop_words='english', max_features=1000)

# # sử dụng TfidfVectorizer để chuyển đổi dữ liệu văn bản thành ma trận TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=200)




# # save df['posts'] to a text file
# df['posts'].to_csv('text.txt', index=False)

df['I-E'] = df['type'].apply(lambda x: 0 if x[0] == 'I' else 1)
df['N-S'] = df['type'].apply(lambda x: 0 if x[1] == 'N' else 1)
df['T-F'] = df['type'].apply(lambda x: 0 if x[2] == 'T' else 1)
df['J-P'] = df['type'].apply(lambda x: 0 if x[3] == 'J' else 1)
# xác định 'I-E' là target và 'posts' là feature
X = vectorizer.fit_transform(df['posts'])

y = df['type']




# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("start training")
start = time.time()


#train a random forest classifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make a prediction on your test data
y_pred = model.predict(X_test)

# Evaluate the model on your training data
acc = model.score(X_train, y_train)
print('Random Forest:', acc)

# Evaluate the model
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print('Random Forest:', acc)



print("time to complete: ", time.time() - start)


# #train a support vector machine classifier
# print("start training")
# start = time.time()
# from sklearn.svm import SVC
# model = SVC()
# model.fit(X_train, y_train)
# # Make a prediction on your test data
# y_pred = model.predict(X_test)
# # Evaluate the model on your training data
# acc = model.score(X_train, y_train)
# print('SVM:', acc)
# # Evaluate the model
# from sklearn.metrics import accuracy_score
# acc = accuracy_score(y_test, y_pred)
# print('SVM:', acc)

# print("time tp compliting: ", time.time() - start)





