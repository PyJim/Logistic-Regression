import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#opening dataset
data = pd.read_csv("Malignant_or_benign.csv")
print(data.head())

#a jointplot of radius_mean and texture_mean
sns.jointplot("radius_mean", "texture_mean", data=data)
plt.show()

#a heatmap of all the data
sns.heatmap(data.corr())
plt.show()

#finding the number of NaNs in the entire dataset... apply print function to show
data.isnull().sum()

#preparing out arguments for the model
X = data[["radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst"]]
#the desired output
y = data["diagnosis"]
#X.head()
#y.head()

#splitting the data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=101)

from sklearn.linear_model import LogisticRegression
logicModel = LogisticRegression()

#training the data to fit the model... can decide to print the output
logicModel.fit(X_train,y_train)

#predicting outputs
y_predict = logicModel.predict(X_test)
print(y_predict)

#comparing actual data to what the model predicted
from sklearn.metrics import classification_report
precision=classification_report(y_test, y_predict)
print(precision)

#printing the confusion
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_predict)
print(confusion)

