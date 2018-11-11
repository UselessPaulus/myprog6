import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
irisData = pd.read_csv("C:/Users/ganch/PycharmProjects/myprog6/venv/iris.csv")

print(irisData.head())
print(irisData.describe())
print(irisData.corr())

features = irisData[["SepalLenght","SepalWidth","PetalLenght","PetalWidth"]]
targetVariables = irisData.Class

featureTrain, featureTest, targetTrain, targetTest = train_test_split(features, targetVariables, test_size=.2)

model = DecisionTreeClassifier()
fittedModel = model.fit(featureTrain, targetTrain)
predictions = fittedModel.predict(featureTest)


print(confusion_matrix(targetTest, predictions))
print(accuracy_score(targetTest, predictions))
