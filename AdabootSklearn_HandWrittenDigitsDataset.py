from sklearn import datasets, utils, ensemble
import pandas as pd
from matplotlib import pyplot as plt

digitsData = datasets.load_digits()

# keys in digitsData:
#           data -> shape (1797,64) pixels: 64 , 1797 images
#           traget -> [0....9] 10 different classes
#           target_names -> [0...9]
#           images -> the images 8 by 8
#           DESCR ->read me

inputs, outputs = utils.shuffle(digitsData.data, digitsData.target, random_state = 511)

features = pd.DataFrame(inputs) #image flattened
targets = pd.DataFrame(outputs)

# let's take 1500 for training and 297 for testing (~85/15 radio)

trainingData = features.loc[0:1500]
testingData = features.loc[1500:]
trainingTarget = targets.loc[0:1500]
testingTarget = targets.loc[1500:]

#Creating a weak calssifier using random forest AdaBoostClassifier

#Random forest Classifier:
# n_estimators = the number of trees in the forest
#criterion default = "gini" --function to measure the quality of a split
#max_depth = the maximum depth of a tree, none = until all leaves are pure


forest = ensemble.RandomForestClassifier(max_depth = 1, n_estimators= 1000)

forest.fit(trainingData,trainingTarget)

weakAccuracyTest = forest.score(testingData,testingTarget) #problem here

tryAdaboost = ensemble.AdaBoostClassifier(ensemble.RandomForestClassifier(max_depth = 1, n_estimators= 1000))

tryAdaboost.fit(trainingData,trainingTarget)

tryAdaboostAccuracyTest = tryAdaboost.score(testingData,testingTarget)

print('The weak classifier has an accuracy of', weakAccuracyTest)

print('The boosted classifier has an accuracy of', tryAdaboostAccuracyTest)
