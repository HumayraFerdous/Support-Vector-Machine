import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv('Breast_Cancer_data.csv')
#print(data.head())
#print(data.shape)
#print(data.isnull().sum())

data.drop(['id','Unnamed: 32'],axis=1,inplace=True)
#print(data.dtypes)
data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})

print(data['diagnosis'].value_counts())
plt.figure(figsize=(6, 4))
sns.countplot(x='diagnosis', data=data,hue='diagnosis',legend = True)
plt.title("Diagnosis Class Distribution (0=Benign, 1=Malignant)")
plt.show()

corr = data.corr()
top_corr_features = corr['diagnosis'].abs().sort_values(ascending=False).head(11).index
plt.figure(figsize=(10, 8))
sns.heatmap(data[top_corr_features].corr(), annot=True, cmap='coolwarm')
plt.title("Top Correlated Features with Diagnosis")
plt.show()

print(top_corr_features)

X = data.drop(['diagnosis'],axis=1)
y = data['diagnosis']

sc = StandardScaler()
X_scaled = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = SVC(kernel = 'linear')
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

feature_importance = np.abs(model.coef_[0])
feature_names = data.drop(columns = ['diagnosis']).columns
important_features = pd.Series(feature_importance,index = feature_names)
important_features = important_features.sort_values(ascending=False)

print("Top 10 important features (linear SVM):")
print(important_features.head(10))

