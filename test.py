from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import re
from collections import Counter

# Load the data
df = pd.read_csv('./data/mbti_1.csv')

# Preprocess the data
df['words_per_comment'] = df['posts'].apply(lambda x: len(x.split())/50)
df['http_per_comment'] = df['posts'].apply(lambda x: x.count('http')/50)
df['music_per_comment'] = df['posts'].apply(lambda x: x.count('music')/50)
df['question_per_comment'] = df['posts'].apply(lambda x: x.count('?')/50)
df['img_per_comment'] = df['posts'].apply(lambda x: x.count('jpg')/50)
df['excl_per_comment'] = df['posts'].apply(lambda x: x.count('!')/50)
df['ellipsis_per_comment'] = df['posts'].apply(lambda x: x.count('...')/50)

# def preprocess_data(data):
#     # xóa http link và https link
#     data = re.sub(r'http\S+', '', data)
#     # xóa các ký tự đặc biệt
#     data = re.sub(r'[“€â.|,?!)(1234567890:/-]', '', data)
#     # xóa các link
#     data = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', data)
#     # xóa các ký tự đặc biệt
#     data = re.sub(r'[|)(?.,:1234567890!]', ' ', data)
#     # xóa các khoảng trắng thừa
#     data = re.sub(' +', ' ', data)
#     return data

#df['raw_posts'] = df['posts'].apply(preprocess_data)


new_column=[]
for z in range(len(df['posts'])):
    prov=df['posts'][z]
    # xóa http link và https link
    prov1 = re.sub(r'http\S+', '', prov)
    prov2= re.sub(r'[“€â.|,?!)(1234567890:/-]', '', prov1)
    prov3 = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', prov2)
    prov4 = re.sub(r'[|)(?.,:1234567890!]',' ',prov3)
    prov5 = re.sub(' +',' ', prov4)
    prov6 = prov5.split(" ")
    counter = Counter(prov6)
    counter2 = counter.most_common(1)[0][0]
    new_column.append(counter2)
df['most_used_word'] = new_column

# # đánh index cho từng từ
word_to_index = {word: i for i, word in enumerate(df['most_used_word'].unique())}
df['most_used_word'] = df['most_used_word'].apply(lambda x: word_to_index[x])





# Split the data into features and labels for each attribute
map1 = {"I": 0, "E": 1}
map2 = {"N": 0, "S": 1}
map3 = {"T": 0, "F": 1}
map4 = {"J": 0, "P": 1}
X = df.drop(['type','posts'], axis=1).values
Y_ie = df['type'].apply(lambda x: map1[x[0]]).values
Y_ns = df['type'].apply(lambda x: map2[x[1]]).values
Y_tf = df['type'].apply(lambda x: map3[x[2]]).values
Y_jp = df['type'].apply(lambda x: map4[x[3]]).values


# Split the data into training and test sets for each attribute
X_train, X_test, Y_ie_train, Y_ie_test = train_test_split(X, Y_ie, test_size=0.2, random_state=42)
X_train, X_test, Y_ns_train, Y_ns_test = train_test_split(X, Y_ns, test_size=0.2, random_state=42)
X_train, X_test, Y_tf_train, Y_tf_test = train_test_split(X, Y_tf, test_size=0.2, random_state=42)
X_train, X_test, Y_jp_train, Y_jp_test = train_test_split(X, Y_jp, test_size=0.2, random_state=42)


# Evaluate the models (accuracy and classification report)
def evaluate_model(model, X_test, Y_test):
    Y_prediction = model.predict(X_test)
    acc = accuracy_score(Y_test, Y_prediction)
    print(acc)


#train the model using Random Forest
random_forest_ie = RandomForestClassifier().fit(X_train, Y_ie_train)
random_forest_ns = RandomForestClassifier().fit(X_train, Y_ns_train)
random_forest_tf = RandomForestClassifier().fit(X_train, Y_tf_train)
random_forest_jp = RandomForestClassifier().fit(X_train, Y_jp_train)

# Evaluate the models
print('Random Forest')
evaluate_model(random_forest_ie, X_test, Y_ie_test)
evaluate_model(random_forest_ns, X_test, Y_ns_test)
evaluate_model(random_forest_tf, X_test, Y_tf_test)
evaluate_model(random_forest_jp, X_test, Y_jp_test)

#train the model using knn
from sklearn.neighbors import KNeighborsClassifier
knn_ie = KNeighborsClassifier().fit(X_train, Y_ie_train)
knn_ns = KNeighborsClassifier().fit(X_train, Y_ns_train)
knn_tf = KNeighborsClassifier().fit(X_train, Y_tf_train)
knn_jp = KNeighborsClassifier().fit(X_train, Y_jp_train)

# Evaluate the models
print('KNN')
evaluate_model(knn_ie, X_test, Y_ie_test)
evaluate_model(knn_ns, X_test, Y_ns_test)
evaluate_model(knn_tf, X_test, Y_tf_test)
evaluate_model(knn_jp, X_test, Y_jp_test)

#train the model using SVM
from sklearn.svm import SVC
svm_ie = SVC().fit(X_train, Y_ie_train)
svm_ns = SVC().fit(X_train, Y_ns_train)
svm_tf = SVC().fit(X_train, Y_tf_train)
svm_jp = SVC().fit(X_train, Y_jp_train)

# Evaluate the models
print('SVM')
evaluate_model(svm_ie, X_test, Y_ie_test)
evaluate_model(svm_ns, X_test, Y_ns_test)
evaluate_model(svm_tf, X_test, Y_tf_test)
evaluate_model(svm_jp, X_test, Y_jp_test)

#train the model using logistic regression
from sklearn.linear_model import LogisticRegression
logistic_regression_ie = LogisticRegression(max_iter=10000).fit(X_train, Y_ie_train)
logistic_regression_ns = LogisticRegression(max_iter=10000).fit(X_train, Y_ns_train)
logistic_regression_tf = LogisticRegression(max_iter=10000).fit(X_train, Y_tf_train)
logistic_regression_jp = LogisticRegression(max_iter=10000).fit(X_train, Y_jp_train)

# Evaluate the models
print('Logistic Regression')
evaluate_model(logistic_regression_ie, X_test, Y_ie_test)
evaluate_model(logistic_regression_ns, X_test, Y_ns_test)
evaluate_model(logistic_regression_tf, X_test, Y_tf_test)
evaluate_model(logistic_regression_jp, X_test, Y_jp_test)





