# Supervised Learning - Prediction of the kind of Fruit by (classification & regression)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import r2_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42) # seed for reproducibility

# 1. Data preparation (Hypothetical data of fruits)
weight = np.random.normal(150, 30, 100) # in grams
volume = 2 * weight + np.random.normal(0, 10, 100) # in cm^3
classes = (weight + volume > 450).astype(int) # define the class(label), heavy fruits in class 1, light fruits class 0.

'''sns.displot(weight)
sns.displot(volume)
sns.displot(classes)
plt.show()'''

data = {
    "weight": weight,
    "volume": volume,
    "classes": classes
}
df = pd.DataFrame(data) # load data into a dataframe object

X = df[['weight', 'volume']] # independent values
y = df['classes'] # dependent value
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2 , random_state=42)


# 2.1 Linear Regression
lin_model = LinearRegression() #create a linear regression object.
lin_model.fit(train_X[['weight']], train_X['volume']) # by [[]] is train_X in 2D
y_pred_lin = lin_model.predict(test_X[['weight']])
score = r2_score(test_X['volume'], y_pred_lin) # actual vs predicted values
#score = lin_model.score(test_X[['weight']], test_X['volume'])

#print("Linear Regression Predicted values:", y_pred_lin[:5])
print("Linear Regression R2:", score) # R^2 Score shows how well the model explains the variance of the data


# 2.2 Logistic Regression
logr_model = LogisticRegression()
logr_model.fit(train_X, train_y)
y_pred_logr = logr_model.predict(test_X)
print(f"Logistic Regression Accuracy: {accuracy_score(test_y, y_pred_logr)}") # (Number of correct predictions) / (Total data)

'''confusion_matrix = metrics.confusion_matrix(test_y, y_pred_logr) # test_y: actual, y_pred: predicted
Accuracy = metrics.accuracy_score(test_y, y_pred_logr) 
print("Logistic Regression Predicted values:", y_pred_logr[:5])
print(metrics.classification_report(test_y, y_pred_logr))
print("Confusion Matrix:\n", confusion_matrix)
print("Accuracy in Logistic Regression:", Accuracy)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1]) # 0, 1, 2 for 3 classes
cm_display.plot()
plt.show()'''


# 2.3 KNN
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)
test_X_scaled = scaler.transform(test_X)

knn_model = KNeighborsClassifier(n_neighbors=3)  # k=3
knn_model.fit(train_X_scaled, train_y)
y_pred_knn = knn_model.predict(test_X_scaled)
print(f"KNN Accuracy: {accuracy_score(test_y, y_pred_knn)}")
#print("KNN Predicted values:", y_pred_knn[:5])

'''plt.scatter(train_X['weight'], train_X['volume'], c=train_y, cmap='bwr', label='Training Data')
plt.scatter(test_X['weight'], test_X['volume'], c=y_pred_knn, marker='x', cmap='bwr', label='Predictions')
plt.xlabel("weight")
plt.ylabel("Volume")
plt.title("KNN Classification")
plt.legend()
plt.show()'''



# 2.4 Decision tree
dtree_model = DecisionTreeClassifier(random_state=42)
dtree_model.fit(train_X, train_y)
y_pred_dtree = dtree_model.predict(test_X)
print(f"Decision Tree Accuracy: {accuracy_score(test_y, y_pred_dtree)}")
#print("Decision tree Predicted values:", y_pred_dtree[:5])