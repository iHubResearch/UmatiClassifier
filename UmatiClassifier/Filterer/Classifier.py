'''
Created on Nov 10, 2013

@author: phcostello
'''

# if __name__ == '__main__':
#     pass
from BoilerPlate import *

import json

#Read data
fin = open('Filterer/Data/annotated_data.json')
data_str = fin.read()
fin.close()
data_json = json.loads(data_str)
df_data = pd.DataFrame(data_json)

df_data.info()
df_data.head()

#This extract text at numpy array
text_np = df_data['post_comments_message']

#Make large word features
#uses sklearn CountVectoriser/bag of words
#This is fitting vocab
from sklearn.feature_extraction.text import CountVectorizer
vectoriser_training = CountVectorizer(min_df=1,stop_words='english',strip_accents='unicode')
t = time.time()
features = vectoriser_training.fit_transform(text_np) 
print "training text to word vector took", time.time()-t, "seconds"
features.shape
#Change from sparse matrix to dense matrix
#vect_train = features_msg_training.todense() #BIG memory usage 2GB, see how to use the sparse for training

#Create target
target = df_data['T.F']
target = target.values
target.shape


#Create classifier
from sklearn.svm import LinearSVC

X= features
y = list(target)

t = time.time()
clf=LinearSVC(C=1)
clf.fit(X,y)
print "training took", time.time()-t, "seconds"

#Check performance
training_predicted = clf.predict(X)
training_predicted

from sklearn import metrics

type(y)
type(training_predicted)
y.shape
training_predicted.shape

cm_training = metrics.confusion_matrix(y,training_predicted)
print "Confusion Matrix on training data"
print cm_training

from sklearn import cross_validation

fin = open('Filterer/Data/allcomments.json')
allcomments = fin.read()
fin.close()
allcomments_str = allcomments
contrl_char = allcomments_str[1864174] #\x03 Ctrl character
allcomments_str =allcomments_str.replace(contrl_char,"")
contrl_char = allcomments_str[17885956] #\x0e
allcomments_str =allcomments_str.replace(contrl_char,"")

#Have to remove weird characters above - control and soemthing else
allcomments_uni = allcomments_str.decode('utf-8')
df_allcomments = pd.DataFrame(json.loads(allcomments_uni))
df_allcomments.info()

text_allcomments = df_allcomments['post_comments_message']
X_full = vectoriser_training.transform(text_allcomments)
X_full.shape
#Check performance
training_predicted = clf.predict(X_full)


y_full_true = [ 'FALSE' for i in range(0,len(training_predicted))]
training_predicted.shape
len(y_full_true)
cm_training = metrics.confusion_matrix(y_full_true,training_predicted)
print "Confusion Matrix on training data"
print cm_training


Xcsr = X.tocsr()
row = Xcsr[0].todense()
row
skf = cross_validation.StratifiedKFold(y=target, n_folds=3)
for train_index, test_index in skf:
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = Xcsr[train_index], Xcsr[test_index]
    y_train, y_test = [y[it] for it in train_index], [y[it] for it in test_index]
    


#Cross validation not straightforward with sparse matrix
#as iterating over k-folds requires X to dense as has to slice
#on ranges. When doing the fitting this kill computer memory

clfs = []
cms = []
labels = []
for train_index, test_index in skf:
    y_train = [y[it] for it in train_index]
    y_test = [y[it] for it in test_index]
    this_clf =clf.fit(Xcsr[train_index],y_train)
    clfs.append(this_clf)
    this_predicted = this_clf.predict(Xcsr[test_index])
    print np.array(zip(y_test,this_predicted))
    
# #         print clf.fit(Xcsr[train_index],y[train_index]).score(Xcsr[test_index],y[test_index])
# #     this_label1 = unique_labels(list(y[test_index]))
# #     this_label2 = unique_labels(list(this_predicted))
#     this_label1 = np.unique(y[test_index]).tolist()
#     this_label2 = np.unique(this_predicted).tolist()
#     this_label = list(set(this_label1).union( set(this_label2)))
#     print this_label
# #     this_label = this_label.tolist()
#     print len(this_label)
#     cm_test = metrics.confusion_matrix(y_test,
#                                        this_predicted)#,this_label) 
#     print cm_test.shape
# #     labels.append(this_label)
#     cms.append(cm_test)
#     
#    


 
