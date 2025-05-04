import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
data = pd.read_csv("bank.csv")
print(data.head())
print(data.isnull().sum())
cat_cols= [col for col in data.columns if data[col].dtype == 'object']
num_cols= [col for col in data.columns if data[col].dtype != 'object']
print(cat_cols)
print(num_cols)

le =  LabelEncoder()
sc = StandardScaler()
for i in cat_cols:
    data[i] = le.fit_transform(data[i])
for i in num_cols:
    data[[i]] = sc.fit_transform(data[[i]])
print(data['deposit'].value_counts())

X = data.drop('deposit',axis=1)
y = data['deposit']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

rbf_svm = SVC(kernel='rbf', C=1.0, gamma='scale')
rbf_svm.fit(X_train, y_train)
y_pred_rbf = rbf_svm.predict(X_test)

print("RBF Kernel SVM Classification Report:")
print(classification_report(y_test, y_pred_rbf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rbf))

# Polynomial Kernel SVM
poly_svm = SVC(kernel='poly', degree=3, C=1.0, gamma='scale')
poly_svm.fit(X_train, y_train)
y_pred_poly = poly_svm.predict(X_test)

print("\nPolynomial Kernel SVM Classification Report:")
print(classification_report(y_test, y_pred_poly))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_poly))

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 'scale'],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)


best_rbf_svm = grid.best_estimator_
y_pred_best = best_rbf_svm.predict(X_test)

print("Best RBF SVM Classification Report:")
print(classification_report(y_test, y_pred_best))
# Since the original data has many features, we reduce to 2 dimensions using PCA just for visualization.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)

svm_vis = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_vis.fit(X_train_pca, y_train_pca)

h = 0.02
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap="coolwarm", alpha=0.3)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='coolwarm', alpha=0.7)
plt.title("SVM Decision Boundary (PCA-Reduced Data)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()