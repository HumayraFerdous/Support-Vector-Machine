import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix

data = pd.read_csv("penguins_size.csv")
#print(data.head())
#print(data.shape)
#print(data.dtypes)
#print(data.isnull().sum())
data['culmen_length_mm'] = data['culmen_length_mm'].fillna(data['culmen_length_mm'].median())
data['culmen_depth_mm'] = data['culmen_depth_mm'].fillna(data['culmen_depth_mm'].median())
data['flipper_length_mm'] = data['flipper_length_mm'].fillna(data['flipper_length_mm'].median())
data['body_mass_g'] = data['body_mass_g'].fillna(data['body_mass_g'].median())

#print(data.isnull().sum())

print(data['species'].value_counts())
data = data[data['species']!='Chinstrap']
encoder = LabelEncoder()
data['species'] = encoder.fit_transform(data['species'])
data['island'] = encoder.fit_transform(data['island'])
data['sex'] = encoder.fit_transform(data['sex'])


X = data.drop(['species'],axis=1).values
y = data['species'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = SVC(kernel='linear',C=1.0)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

