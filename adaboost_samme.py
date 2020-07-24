import numpy as np
import pandas as pd
import math
from sklearn import tree
from sklearn.metrics import accuracy_score

class BoostedDT:

    def __init__(self, numBoostingIters=100, maxTreeDepth=3):
        '''
        clfs : List object containing individual DecisionTree classifiers, in order of creation during boosting
        betas : List of beta values, in order of creation during boosting
        '''
        self.clfs = []
        self.betas = []
        self.numBoostingIters = numBoostingIters
        self.maxTreeDepth = maxTreeDepth
        self.k = 0
        self.classes = []



    def fit(self, X, y, random_state=None):
        '''
        Arguments:
            X is an n-by-d Pandas Data Frame
            y is an n-by-1 Pandas Data Frame
            random_seed is an optional integer value
        '''
        self.models = []
        self.classes = np.unique((y))
        self.K = len(self.classes)
        n,d = X.shape
        weight = np.full((n,),1/n)
        for i in range(self.numBoostingIters):
          clf = DecisionTreeClassifier(max_depth=self.maxTreeDepth).fit(X,y,sample_weight=weight)
          prediction = clf.predict(X)
          e = 1 - accuracy_score(y, prediction, sample_weight=weight)
          beta = np.log((1-e)/e) + np.log(self.K - 1)
          match = prediction==y
          weight[~match] *= np.exp(beta)
          weight /= weight.sum()
          self.clfs.append(clf)
          self.betas.append(beta)



    def predict(self, X):
        '''
        Arguments:
            X is an n-by-d Pandas Data Frame
        Returns:
            an n-by-1 Pandas Data Frame of the predictions
        '''
        n = len(X)
        pred = np.zeros((n,self.K))
        i = 0
        for beta,clf in zip(self.betas, self.clfs):
            yp = clf.predict(X).astype(int)
            pred[range(n),yp] += beta
            i += 1
        pred = np.argmax(pred,axis=1)
        return pred
