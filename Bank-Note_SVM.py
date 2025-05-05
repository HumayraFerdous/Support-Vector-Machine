import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt'
columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
df = pd.read_csv(url, header=None, names=columns)
#print(df.head())
#print(df.dtypes)
#print(df.isnull().sum())

X = df.drop('class',axis=1)
y = df['class']

sns.countplot(x='class',data=df)
plt.title("Class Distribution (0 = Genuine, 1 = Forged")
plt.show()

sns.pairplot(df, hue='class')
plt.suptitle("Pairwise Feature Plots", y=1.02)
plt.show()

corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

features = ['variance', 'skewness', 'curtosis', 'entropy']
for feature in features:
    sns.kdeplot(data=df, x=feature, hue='class', fill=True)
    plt.title(f"{feature.capitalize()} Distribution by Class")
    plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_poly = SVC(kernel='poly', degree=3, C=1.0, gamma='scale')
svm_poly.fit(X_train_scaled, y_train)
y_pred = svm_poly.predict(X_test_scaled)
print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

param_grid = {
    'degree': [2, 3, 4],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(SVC(kernel='poly'), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)
print("Best Parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)
print("\nClassification Report for Best Model:")
print(classification_report(y_test, y_pred_best))