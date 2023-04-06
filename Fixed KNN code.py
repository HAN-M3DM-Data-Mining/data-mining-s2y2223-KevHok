# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 20:54:07 2023

@author: hoksb
"""
#errors found:
#Line 10: was import pandas as pandas 
#is import pandas as pd

#Line 15: was from sklearn.metrics import accuracy_scores 
# is from sklearn.metrics import accuracy_score

#Line 19: was rawData = pd.read_xslx(URL) 
# is rawData = pd.read_csv(URL)

#Line 54: def normalize(x):
#was   return (((max)x - min(x)) / (max(x) - min(x)))
              # function to normalize

#is def normalize(x):
#    return (((x) - min(x)) / (max(x) - min(x)))
              # function to normalize
              
#Line 43: was #prepData.info()
# is prepData.info()

#Library import

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
#Data import

URL = "https://raw.githubusercontent.com/HAN-M3DM-Data-Mining/data-mining-s2y2223-robindepagter/master/datasets/KNN-diabetes.csv"
rawData = pd.read_csv(URL)
rawData.info()
#Data Prep

# As far as the dataset goes, there are no columns that can not be used for predicting diabetes, in other words: nothing has to be eliminated from the dataset.
prepData = rawData
prepData.head()

# The column 'Outcome' is what we want to predict, 1 is tested and diagnosed with Diabetes and 0 is tested but not diagnosed with diabetes.
cntOutcome = prepData['Outcome'].value_counts()
normalizedOutcome = prepData['Outcome'].value_counts(normalize=True)

# Amount of Outcomes
print("Outcomes \n",cntOutcome)

# Normalized amount of Outcomes
print("\n Normalized outcomes \n", normalizedOutcome)

# Transform the column 'Outcome' to the type Category instead of integer.
# This is necessary since most models in general cannot use an Integer as a category, they require a column of the type 'Category
catType = pd.CategoricalDtype(categories=[0, 1], ordered=False)
prepData['Outcome'] = prepData['Outcome'].astype(catType)

# To see if it worked, uncomment the follow line of code:
prepData.info()

#### IMPORTANT INFORMATION####
# To use our data for the model we must first prepare it.
# For testing purposes different preparing methods have been used like robust and standard scaling.
# None of which increased the accuracy of the KNN model above a normalized dataset.
# Normalizing data will prevent certain broader ranges of data to negatively influence our outcomes.
# To normalize our data we will create a normalizing function


def normalize(x):
    return (((x) - min(x)) / (max(x) - min(x)))
              # function to normalize


excluded = ['Outcome']  # This column will be excluded
# This will fetch the data minus the excluded column and put it into the variable X
x = prepData.loc[:, ~prepData.columns.isin(excluded)]
# This will drag the variable X through the normalize function
x = x.apply(normalize, axis=0)
print(x[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']
  ].describe())  # This is the same as earlier code but now normalized

# for the final preparation we will split our data into two group: Test and Train using the following function:
y = prepData['Outcome']
# Test and train will be 25/75 percent of the data. Stratify: ratio yes and no are equal in test and train set
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=123, stratify=y)

##################################
# DATA PREPARATION IS NOW COMPLETE#
##################################
#Modelling and evaluation

# After the preparation we can finally begin to model and execute the code. For k-Nearest Neighbour we can use the following code:

# Model in Pseudocode:
#  for each instance in the test set:
#     for each instance in the training set:
#         calculate the distance between the two instances
#     sort the distances in ascending order
#     find the K nearest neighbors
#     predict the class based on the majority class among the K nearest neighbors

# Model in python code:
knn = KNeighborsClassifier(n_neighbors=5)
# This means that it will classify the data using it's 5 nearest neighbours.
# So in other words: I have 5 neighbours with similar data, what is their classification? If most or all, for example, are category 1, it will apply category 1.

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
cm

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot()

plt.show()

# Check performance using accuracy - Accuracy score of the model
print("Accuracy: ", accuracy_score(y_test, y_pred))

# If we take a look at our diffusion table, we can see that the model generated a lot of false negatives which could indicate that our model was not extremely accurate.
# The code however indicates that our model was about 78% accurate which is probably, in the context of the model, not accurate enough to be usable in practice.

# The many false negatives can be potentially dangerous, if a patient is told "Nothing to worry about, you don't have diabetes" and he/she in fact does have diabetes.
# It could be putting the patient at an unnecessary high risk.