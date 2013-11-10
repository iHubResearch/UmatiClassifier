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
floats = []
for id, it in enumerate(text.tolist()):
    if isinstance(it, float):
        floats.append(id)
floats

unicode(text[9])


#Make large word features
#uses sklearn CountVectoriser/bag of words
#This is fitting vocab
from sklearn.feature_extraction.text import CountVectorizer
vectoriser_training = CountVectorizer(min_df=1,stop_words='english',strip_accents='unicode')
t = time.time()
features_msg_training = vectoriser_training.fit_transform(text) 
print "training text to word vector took", time.time()-t, "seconds"
features_msg_training.shape
#Change from sparse matrix to dense matrix
#vect_train = features_msg_training.todense() #BIG memory usage 2GB, see how to use the sparse for training


# Create classifiers

# Use n-fold validation to check which is best.