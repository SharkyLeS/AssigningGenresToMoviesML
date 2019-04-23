import sklearn
import skmultilearn
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn import *
from skmultilearn import adapt
from skmultilearn import problem_transform
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#Store classifiers in a table to change easily from one to another later on
# classifiers = [
#     KNeighborsClassifier(5),
#     SVC(gamma='scale', C=10),
#     GaussianProcessClassifier(1.0 * RBF(1.0)),
#     DecisionTreeClassifier(max_depth=5),
#     RandomForestClassifier(max_depth=10, n_estimators=20, max_features=2),
#     MLPClassifier(alpha=1),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     QuadraticDiscriminantAnalysis()]

#Initiate time measure
startTime = time.perf_counter()

#Load train set
data_set=sklearn.datasets.load_files("Data", categories='movies')

data_set.data[0]=data_set.data[0].decode('utf8')

#Reshape because it is initially just one block of text : we separate it on '\n'
data_set_reshaped=data_set.data[0].split('\n')

#Remove last element which is an empty list
data_set_reshaped.pop()

#Remove labels row
del data_set_reshaped[0]

#Separating the features from the string and cleaning entries while storing genres separately
target=[]
for i in range(len(data_set_reshaped)):
    data_set_reshaped[i]=data_set_reshaped[i].split('"')
    title = data_set_reshaped[i][1]
    titleCut = title.split(',')
    date = titleCut[1]
    nameMov = titleCut[0]
    data_set_reshaped[i][1]=nameMov
    data_set_reshaped[i][0]=date
    target.append(data_set_reshaped[i].pop().replace(',','').split('|'))
    for j in range(len(data_set_reshaped[i])):
        data_set_reshaped[i][j].replace(',','')

#Create encoder and fit it to the dataset
encoder = sklearn.preprocessing.OneHotEncoder()
encoder.fit(data_set_reshaped)
multilabelencoder = sklearn.preprocessing.MultiLabelBinarizer()
multilabelencoder.fit(target)

#Reinsert genres into dataset before train/test separation
for i in range(len(data_set_reshaped)):
    data_set_reshaped[i].append(target[i])

#Create train and test sets
train_set,test_set = sklearn.model_selection.train_test_split(data_set_reshaped, test_size=0.1)

#Extracting date from the movie title and create train and test targets
train_target = []
for i in range(len(train_set)):
    train_target.append(train_set[i].pop())

test_target = []
for j in range(len(test_set)):
    test_target.append(test_set[j].pop())

#transform data from string to numerical data that can be processed by classification algorithms
train_set = encoder.transform(train_set)
test_set = encoder.transform(test_set)
train_target = multilabelencoder.transform(train_target)
test_target = multilabelencoder.transform(test_target)

#Create classifier and train

#tuned_parameters = [{'k':[1,2,3,5,10,20,30,50,100]}]
#clf = sklearn.model_selection.GridSearchCV(skmultilearn.adapt.MLkNN(),tuned_parameters,cv=5)
#clf=sklearn.svm.SVC(gamma='scale')

clf=skmultilearn.adapt.MLkNN(k=5)
clf.fit(train_set,train_target)
prediction = clf.predict(test_set)
score = clf.score(test_set,test_target)

#print(clf.best_params_,clf.best_score_)

#Calculate precision of the algorithm

#Uncomment to print the result given by the score() method

#print("Result of the score() method: "+str(score))

labelEncoder = sklearn.preprocessing.LabelEncoder()
x= encoder.inverse_transform(test_set)[:,0]
y=multilabelencoder.inverse_transform(prediction)

newy=[]
newx=[]
for i in range(len(y)):
    for j in range(len(y[i])):
        newx.append(x[i])
        newy.append(y[i][j])

newx=np.asarray(newx)
newy=np.asarray(newy)

#Access extrema dates
beginDate=min(encoder.inverse_transform(test_set)[:,0])
endDate=max(encoder.inverse_transform(test_set)[:,0])

newx=labelEncoder.fit_transform(newx)
newy=labelEncoder.fit_transform(newy)

#Reshape x for prettier figure xlabel
newx+=int(beginDate)

#Uncomment to print overall time
overallTime = time.perf_counter()-startTime
#print("Overall processing time: "+str(overallTime))

#Print results
plt.scatter(newx,newy,c=newy)
plt.title('Movies genre evolution between '+beginDate+' and '+endDate)
plt.ylabel("Genres")
plt.xlabel("Years")
plt.show()