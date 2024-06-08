import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats

df = pd.read_csv('./data/mbti_1.csv')

df['words_per_comment'] = df['posts'].apply(lambda x: len(x.split())/50)
print(df.head(10))
print("*"*40)
print(df.info())

plt.figure(figsize=(15,10))
sns.violinplot(x='type', y='words_per_comment', data=df, inner=None, color='lightgray')
sns.stripplot(x='type', y='words_per_comment', data=df, size=4, jitter=True)
plt.ylabel('Words per comment')
plt.show()




df['http_per_comment'] = df['posts'].apply(lambda x: x.count('http')/50)
df['music_per_comment'] = df['posts'].apply(lambda x: x.count('music')/50)
df['question_per_comment'] = df['posts'].apply(lambda x: x.count('?')/50)
df['img_per_comment'] = df['posts'].apply(lambda x: x.count('jpg')/50)
df['excl_per_comment'] = df['posts'].apply(lambda x: x.count('!')/50)
df['ellipsis_per_comment'] = df['posts'].apply(lambda x: x.count('...')/50)


# g = sns.jointplot(x='words_per_comment', y='ellipsis_per_comment', data=df, kind='kde')
# # Tính toán hệ số tương quan Pearson và giá trị p
# r, p = scipy.stats.pearsonr(df['words_per_comment'], df['ellipsis_per_comment'])
# # Thêm hệ số tương quan Pearson và giá trị p vào biểu đồ
# g.ax_joint.annotate('pearsonr={:.2f}, p={:.2g}'.format(r, p),
#                     xy=(0.1, 0.9), xycoords='axes fraction')
# plt.show()


# # Phân loại dữ liệu theo từng loại
# i = df['type'].unique()
# k = 0
# for m in range(0,2):
#     for n in range(0,6):
#         df_2 = df[df['type'] == i[k]]
#         # sns.jointplot(x='words_per_comment', y='ellipsis_per_comment', data=df_2, kind="hex")
#         g = sns.jointplot(x='words_per_comment', y='ellipsis_per_comment', data=df_2, kind='hex')
#         # Tính toán hệ số tương quan Pearson và giá trị p
#         r, p = scipy.stats.pearsonr(df_2['words_per_comment'], df_2['ellipsis_per_comment'])
#         # Thêm hệ số tương quan Pearson và giá trị p vào biểu đồ
#         g.ax_joint.annotate('pearsonr={:.2f}, p={:.2g}'.format(r, p),
#                             xy=(0.1, 0.9), xycoords='axes fraction')
#         plt.title(i[k],y=0.5)
#         k+=1
#         plt.show()



# new_column=[]
# for z in range(len(df['posts'])):
#     prov=df['posts'][z]
#     # xóa http link và https link
#     prov1 = re.sub(r'http\S+', '', prov)
#     prov2= re.sub(r'[“€â.|,?!)(1234567890:/-]', '', prov1)
#     prov3 = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', prov2)
#     prov4 = re.sub(r'[|)(?.,:1234567890!]',' ',prov3)
#     prov5 = re.sub(' +',' ', prov4)
#     prov6 = prov5.split(" ")
#     counter = Counter(prov6)
#     counter2 = counter.most_common(1)[0][0]
#     new_column.append(counter2)
# df['most_used_word'] = new_column
# print(df['most_used_word'].unique())
# # # đánh index cho từng từ
# word_to_index = {word: i for i, word in enumerate(df['most_used_word'].unique())}
# df['most_used_word'] = df['most_used_word'].apply(lambda x: word_to_index[x])


map1 = {"I": 0, "E": 1}
map2 = {"N": 0, "S": 1}
map3 = {"T": 0, "F": 1}
map4 = {"J": 0, "P": 1}
df['I-E'] = df['type'].astype(str).str[0]
df['I-E'] = df['I-E'].map(map1)
df['N-S'] = df['type'].astype(str).str[1]
df['N-S'] = df['N-S'].map(map2)
df['T-F'] = df['type'].astype(str).str[2]
df['T-F'] = df['T-F'].map(map3)
df['J-P'] = df['type'].astype(str).str[3]
df['J-P'] = df['J-P'].map(map4)

# Evaluate the models (accuracy and classification report)
def evaluate_model(model, X_test, Y_test):
    Y_prediction = model.predict(X_test)
    acc = accuracy_score(Y_test, Y_prediction)
    print(acc)

X = df.drop(['type','posts', 'I-E', 'N-S', 'T-F', 'J-P'], axis=1).values
y = df['J-P'].values

print(y.shape)
print(X.shape)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.2, random_state=42)



# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
#biểu diễn độ quan trọng của các feature
importances = pd.DataFrame({'feature':df.drop(['type','posts', 'I-E', 'N-S', 'T-F', 'J-P'], axis=1).columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
print(importances)

Y_prediction = random_forest.predict(X_test)
# độ chính xác của model trên tập train
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
print(acc_random_forest)
# độ chính xác của model trên tập test
acc = accuracy_score(y_test, Y_prediction)
print(acc)



# from sklearn.model_selection import cross_val_score

# # Random Forest
# random_forest = RandomForestClassifier(n_estimators=100)

# # Sử dụng cross-validation với 5 folds
# scores = cross_val_score(random_forest, X, y, cv=5)

# print("Scores:", scores)
# print("Mean:", scores.mean())
# print("Standard Deviation:", scores.std())



# from sklearn.svm import SVC
# # Khởi tạo mô hình SVM
# model = SVC()
# # Tham số cần tối ưu hóa
# parameters = {'kernel': 'rbf', 'C': 0.1, 'gamma': 1 }
# # tao model với tham số trên
# model = SVC(**parameters)
# # train model
# model.fit(X_train, y_train)
# # dự đoán
# y_pred = model.predict(X_test)
# # đánh giá model
# acc = accuracy_score(y_test, y_pred)
# print(acc)





