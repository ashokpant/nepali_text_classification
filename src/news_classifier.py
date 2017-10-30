# import sklearn
import nltk
import scipy as sp
import numpy as np
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection._search import GridSearchCV
from sklearn.model_selection._split import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# CorpusDirTrain = r'E:\Datasets\16NepaliNews\16NepaliNews\Train'
# CorpusDirTest = r'E:\Datasets\16NepaliNews\16NepaliNews\Test'
CorpusDir = r'data/16NepaliNews/raw/'
raw = load_files(CorpusDir, description=None,
                 load_content=True,
                 encoding='utf-8',
                 decode_error='ignore')

''' Nepali Stop Words '''
# The stop words file is copied into the stopwords directory of nltk.data\corpora folder

stopWords = set(nltk.corpus.stopwords.words('nepali'))

''' Testing and Training Data '''
xTrain, xTest, yTrain, yTest = train_test_split(raw.data,
                                                raw.target,
                                                test_size=0.3,
                                                random_state=42)

''' feature vector construction '''
''' Vectorizer '''

tfidfVectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(" "),
                                  sublinear_tf=True, encoding='utf-8',
                                  decode_error='ignore',
                                  max_df=0.5,
                                  min_df=10,
                                  stop_words=stopWords)

xTrainVec = tfidfVectorizer.fit_transform(xTrain)
print('No of Samples , No. of Features ', xTrainVec.shape)
''' Classifier '''

# Multinomial Naive Bayes
clf1 = Pipeline([
    ('vect', tfidfVectorizer),
    ('clf', MultinomialNB(alpha=0.01, fit_prior=True))
])

# SVM Linear Kernel
clf2 = Pipeline([
    ('vect', tfidfVectorizer),
    ('clf', SVC(kernel='linear', random_state=42, verbose=False, C=2.5))
])
# SVM RBF Kernel
clf3 = Pipeline([
    ('vect', tfidfVectorizer),
    ('clf', SVC(kernel='rbf', random_state=42, verbose=False, C=22022)) # gamma=0.001
])

# C_range = np.logspace(0.1, 10, 50)
# gamma_range = np.logspace(-9, 3, 13)
#
# # C_range = [0.01, 0.1,0.5,1,1.5,2,2.5,3,4,5,6,7,8,9,10,11,12,50,100,500, 1000]
# # gamma_range = [0.001, 0.01, 0.1, 2]
#
# param_grid = dict(gamma=gamma_range)
# cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
# grid = GridSearchCV(SVC(kernel='rbf',C= 22022.019499873735), param_grid=param_grid, cv=2,verbose=10,n_jobs=6)
#
# grid.fit(xTrainVec, yTrain)
#
# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))


# MLP Neural Network
clf4 = Pipeline([
    ('vect', tfidfVectorizer),
    ('clf', MLPClassifier(solver='adam', alpha=1e-5, learning_rate='adaptive', learning_rate_init=0.001,
                          hidden_layer_sizes=(128, 64), random_state=42, verbose=False))
])


def trainAndEvaluate(clf, xTrain, xTest, yTrain, yTest):
    clf.fit(xTrain, yTrain)
    print("Accuracy on training Set : ")
    print(clf.score(xTrain, yTrain))
    print("Accuracy on Testing Set : ")
    print(clf.score(xTest, yTest))
    yPred = clf.predict(xTest)

    print("Classification Report : ")
    print(metrics.classification_report(yTest, yPred))
    print("Confusion Matrix : ")
    print(metrics.confusion_matrix(yTest, yPred))


print('Multinominal Naive Bayes \n')
trainAndEvaluate(clf1, xTrain, xTest, yTrain, yTest)
print('Linear SVM \n')
trainAndEvaluate(clf2, xTrain, xTest, yTrain, yTest)
print('RBF SVM \n')
trainAndEvaluate(clf3, xTrain, xTest, yTrain, yTest)
print('MLP Neural Network')
trainAndEvaluate(clf4, xTrain, xTest, yTrain, yTest)
