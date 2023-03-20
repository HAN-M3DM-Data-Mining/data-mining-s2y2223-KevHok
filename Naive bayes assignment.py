#title: "Assigment - Naive Bayes DIY"
#author:
#  - Kevin Hoksbergen - Author
#  - Robin de pagter - Reviewer
#date: "`r format(Sys.time(), '%d %B, %Y')`"
#output:
#   html_notebook:
 #   toc: true
  #  toc_depth: 2
#---

#```{r}
#library(tidyverse)
#library(tm)
#library(caret)
#library(wordcloud)
#library(e1071)
#```

#Choose a suitable dataset from [this](https://github.com/HAN-M3DM-Data-Mining/assignments/tree/master/datasets) folder and train your own Naive Bayes model. Follow all the steps from the CRISP-DM model.

## Business Understanding
#text and code here

## Data Understanding
#text and code here

## Data Preparation
#text and code here

## Modeling
#text and code here

## Evaluation and Deployment
#text and code here

#reviewer adds suggestions for improving the model
from matplotlib import colors 
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

url = "https://raw.githubusercontent.com/HAN-M3DM-Data-Mining/data-mining-s2y2223-KevHok/master/datasets/NB-fakenews.csv"
rawDF = pd.read_csv(url)
rawDF.head()

# rawDF.replace(to_replace = 0, value="real", inplace=True)
# rawDF.replace(to_replace = 1, value="fake", inplace=True)

catType = pd.CategoricalDtype(categories=[0, 1], ordered=False)
prepData = rawDF.fillna('')
prepData.type = prepData['label'].astype(catType)
prepData.info()


prepData.value_counts()

prepData.value_counts(normalize=True)

realNews = ' '.join([str(prepData[prepData.type==0]['text'])])
fakeNews = ' '.join([str(prepData[prepData.type==1]['text'])])

colorListReal=['#e9f6fb','#92d2ed','#2195c5']
colorListFake=['#f9ebeb','#d57676','#b03636']
colormapReal=colors.ListedColormap(colorListReal)
colormapFake=colors.ListedColormap(colorListFake)
wordcloudReal = WordCloud(background_color='white', colormap=colormapReal).generate(realNews)
wordcloudFake = WordCloud(background_color='white', colormap=colormapFake).generate(fakeNews)

# Display the generated image:
# the matplotlib way:
fig, (wc1, wc2) = plt.subplots(1, 2)
fig.suptitle('Wordclouds for fake news')
wc1.imshow(wordcloudReal)
wc2.imshow(wordcloudFake)
plt.show()

vectorizer = TfidfVectorizer(max_features=1000)
vectors = vectorizer.fit_transform(prepData.text)
wordsDF = pd.DataFrame(vectors.toarray(), columns=vectorizer.get_feature_names_out())
print(wordsDF.head())

xTrain, xTest, yTrain, yTest = train_test_split(wordsDF, prepData.type)

bayes = MultinomialNB()
bayes.fit(xTrain, yTrain)

yPred = bayes.predict(xTest)
yTrue = yTest

accuracyScore = accuracy_score(yTrue, yPred)
print(f'Accuracy: {accuracyScore}')

matrix = confusion_matrix(yTrue, yPred)
labelNames = pd.Series(['0', '1'])
betterMatrix = pd.DataFrame(matrix, columns='Predicted ' + labelNames, index='Is ' + labelNames)
print(betterMatrix)



