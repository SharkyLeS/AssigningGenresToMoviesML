import sklearn
import time
import matplotlib.pyplot as plt
from sklearn import *

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
    target.append(data_set_reshaped[i].pop())
    for j in range(len(data_set_reshaped[i])):
        data_set_reshaped[i][j].replace(',','')

#Create encoder and fit it to the dataset
encoder = sklearn.preprocessing.OneHotEncoder()
encoder.fit(data_set_reshaped)

#Reinsert genres into dataset before train/test separation
for i in range(len(data_set_reshaped)):
    data_set_reshaped[i].append(target[i])

#Create train and test sets
train_set,test_set = sklearn.model_selection.train_test_split(data_set_reshaped, test_size=0.2)

#Extracting date from the movie title and creating train_target array
train_target=[]

entryDuplicas = []
genresDuplicas = []

#Create train_target set by separating genres related to each movie
for i in range(len(train_set)):
    genres = train_set[i].pop().replace(',','').split('|')
    if(len(genres)>1):
        for j in range(1,len(genres)):
            entryDuplicas.append(train_set[i])
            genresDuplicas.append(genres[j])

    train_target.append(genres[0])

#Same for the testing side

entryDuplicas = []
genresDuplicas = []
test_target = []

for i in range(len(test_set)):
    genres = test_set[i].pop().replace(',','').split('|')
    if(len(genres)>1):
        for j in range(1,len(genres)):
            entryDuplicas.append(test_set[i])
            genresDuplicas.append(genres[j])

    test_target.append(genres[0])

#Add duplicate entries to the train set and their related genres to the train_target array
for k in range(len(entryDuplicas)):
    train_set.append(entryDuplicas[k])
    train_target.append(genresDuplicas[k])

#transform data from string to numerical data that can be processed by classification algorithms
train_set = encoder.transform(train_set)
test_set = encoder.transform(test_set)

#Create classifier and train
clf = sklearn.svm.SVC(gamma='scale',C=10)
clf.fit(train_set,train_target)
prediction = clf.predict(test_set)
score = clf.score(test_set,test_target)

#Calculate overall time for algorithm
overallTime = time.perf_counter()-startTime

#Uncomment to print the overall processing time and the result given by the scoring method.

#print("Overall processing time: "+str(overallTime))
#print("Performance given by score(): "+str(score))

#Uncomment to print the five first entries of the test set and the prediction

#print(encoder.inverse_transform(test_set)[:5])
#print(prediction[:5])

labelEncoder = sklearn.preprocessing.LabelEncoder()

x=encoder.inverse_transform(test_set)[:,0]
y=prediction
z=test_target

#Access extrema dates
beginDate=min(x)
endDate=max(x)

x=labelEncoder.fit_transform(x)
y=labelEncoder.fit_transform(y)
z=labelEncoder.fit_transform(z)

#Reshape x for prettier figure xlabel
x+=int(beginDate)

#Print results
plt.scatter(x,y,c=y,label='prediction')
plt.scatter(x,z,c=z,edgecolors='black',label='expected')
plt.title('Movies genre evolution between '+beginDate+' and '+endDate)
plt.ylabel("Genres")
plt.xlabel("Years")
plt.legend()
plt.show()

#Choose best parameters
#Compare several classifiers
#multi-label processing
#Print some results
#Write guide and commentaries in the code