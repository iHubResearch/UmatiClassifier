'''
Created on Nov 10, 2013

@author: phcostello
'''

# if __name__ == '__main__':
#     pass
from BoilerPlate import *

# Import data
rawdata = pd.read_csv('20131107_UmatiData.csv')
rawdata = pd.read_csv('20131107_UmatiData.txt',sep='\t')

from pandas.io.parsers import  ExcelFile
xls = ExcelFile('20131107_UmatiData.xls')
rawdata = xls.parse('20131107_UmatiData', index_col=None, na_values=['NA'])

#drop na actual text rows

rawdata = rawdata.dropna(subset=['Actual text'])

# Extract features
text = rawdata['Actual text']
# floats = []
# for id, it in enumerate(text.tolist()):
#     if isinstance(it, float):
#         floats.append(id)
# floats


#Make large word features
#uses sklearn CountVectoriser/bag of words
#This is fitting vocab
from sklearn.feature_extraction.text import CountVectorizer
vectoriser_training = CountVectorizer(min_df=1,stop_words='english',strip_accents='unicode')
t = time.time()
features = vectoriser_training.fit_transform(text) 
print "training text to word vector took", time.time()-t, "seconds"
features.shape
#Change from sparse matrix to dense matrix
#vect_train = features_msg_training.todense() #BIG memory usage 2GB, see how to use the sparse for training

#Create target
target = rawdata['The text /article can be seen as encouraging the audience to']
target = target.values
target.shape

#Create classifier
from sklearn.svm import LinearSVC
X= features
y = target

t = time.time()
clf=LinearSVC(C=1,penalty ='l1',dual=False).fit(X,y)
print "training took", time.time()-t, "seconds"

#Check performance
training_predicted = clf.predict(X)

training_predicted

from sklearn import metrics
cm_training = metrics.confusion_matrix(y,training_predicted)
print "Confusion Matrix on training data"
print cm_training

from sklearn import cross_validation
y
Xcsr = X.tocsr()
row = Xcsr[0].todense()
skf = cross_validation.StratifiedKFold(y=y, n_folds=3)
for train_index, test_index in skf:
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = Xcsr[train_index], Xcsr[test_index]
    y_train, y_test = y[train_index], y[test_index]


#Cross validation not straightforward with sparse matrix
#as iterating over k-folds requires X to dense as has to slice
#on ranges. When doing the fitting this kill computer memory

clfs = []
for train_index, test_index in skf:
    clfs.append(clf.fit(Xcsr[train_index],y[train_index]))
#      print clf.fit(Xcsr[train_index],y[train_index]).score(Xcsr[test_index],y[test_index])


this_predicted = clfs[1].predict(X)
cm_training = metrics.confusion_matrix(y,this_predicted)
print "Confusion Matrix on training data"
print cm_training.shape



df=pd.DataFrame(target)
df['predicted']=this_predicted
df.to_csv('output.csv')