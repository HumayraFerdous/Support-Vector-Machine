import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

data = pd.read_csv('diabetes.csv')
#print(data.head())
#print(data.dtypes)
#print(data.shape)
#print(data.isnull().sum())
print(data['Outcome'].value_counts())

data.hist(figsize = (10,8))
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

X = data.drop('Outcome',axis =1)
y = data['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.3,random_state=42)
model = SVC(kernel='rbf', C=1.0, gamma='scale',class_weight='balanced')
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
model2 = SVC(kernel='rbf', C=1.0, gamma='scale')
model2.fit(X_resampled, y_resampled)
y_pred2 = model2.predict(X_test)
print("\nClassification Report after SMOTE:\n", classification_report(y_test, y_pred2))

data2 = data
cols_with_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data2[cols_with_missing] = data2[cols_with_missing].replace(0, np.nan)
data2[cols_with_missing] = data2[cols_with_missing].fillna(data2[cols_with_missing].median())

X2 = data2.drop('Outcome',axis =1)
y2 = data2['Outcome']

scaler = StandardScaler()
X_rescaled = scaler.fit_transform(X2)

X2_train,X2_test,y2_train,y2_test = train_test_split(X_rescaled,y2,test_size=0.3,random_state=42)
model2 = SVC(kernel='rbf', C=1.0, gamma='scale')
model2.fit(X2_train, y2_train)

y2_pred = model.predict(X2_test)
print("\nClassification Report:\n", classification_report(y2_test, y2_pred))

