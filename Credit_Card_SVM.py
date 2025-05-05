import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv("creditcard.csv")
#print(data.head())
#print(data.dtypes)
#print(data.isnull().sum())
#print(data['Class'].value_counts())

"""data_fraud = data [data['Class'] == 1]
plt.figure(figsize = (15,10))
plt.scatter(data_fraud['Time'],data_fraud['Amount'])
plt.title('Scatter plot amount fraud')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.xlim([0,175000])
plt.ylim([0,2500])
plt.show()"""

data_corr = data.corr()
#plt.figure(figsize=(15,10))
#sns.heatmap(data_corr, cmap="YlGnBu") # Displaying the Heatmap
#sns.set_theme(font_scale=2,style='white')
#plt.title('Heatmap correlation')
#plt.show()

rank = data_corr['Class']
df_rank = pd.DataFrame(rank)
df_rank = np.abs(df_rank).sort_values(by='Class',ascending=False)
df_rank.dropna(inplace=True)
#print(df_rank)

df_train_all = data[0:150000]
df_train_1 = df_train_all[df_train_all['Class'] == 1]
df_train_0 = df_train_all[df_train_all['Class'] == 0]
print('In this dataset, we have ' + str(len(df_train_1)) +" frauds so we need to take a similar number of non-fraud")

df_sample=df_train_0.sample(300)
df_train = df_train_1._append(df_sample)
df_train = df_train.sample(frac=1)

X_train = df_train.drop(['Time', 'Class'],axis=1)
y_train = df_train['Class']
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

df_test_all = data[150000:]

X_test_all = df_test_all.drop(['Time', 'Class'],axis=1)
y_test_all = df_test_all['Class']
X_test_all = np.asarray(X_test_all)
y_test_all = np.asarray(y_test_all)

X_train_rank = df_train[df_rank.index[1:11]]
X_train_rank = np.asarray(X_train_rank)
X_test_all_rank = df_test_all[df_rank.index[1:11]]
X_test_all_rank = np.asarray(X_test_all_rank)
y_test_all = np.asarray(y_test_all)

classifier = SVC(kernel='linear',C=1.0)
classifier.fit(X_train,y_train)
prediction_SVM_all = classifier.predict(X_test_all)
cm = confusion_matrix(y_test_all, prediction_SVM_all)
print('Confusion matrix: \n',cm)
print("Classification Report: \n",classification_report(y_test_all,prediction_SVM_all))

"""classifier.fit(X_train_rank, y_train)
prediction_SVM = classifier.predict(X_test_all_rank)
cm = confusion_matrix(y_test_all, prediction_SVM)
print('Confusion matrix: \n',cm)
print("Classification Report: \n",classification_report(y_test_all,prediction_SVM_all))"""
