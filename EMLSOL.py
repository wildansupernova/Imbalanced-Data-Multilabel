from MLSOL import MLSOL
from sklearn.metrics import f1_score

import copy
import numpy as np

class EMLSOL:
    def __init__(self, baseMultiLabelLearner = None, mlSampling = MLSOL(), numModels = 5, samplingRatio = 0.3, randomState = 1):
        self.baseMultiLabelLearner = baseMultiLabelLearner
        self.mlSampling = mlSampling
        self.numModels = numModels
        self.samplingRatio = samplingRatio
        self.randomState = randomState
        self.thresholds = None

    def fit(self, X, y):
        self.mlls = []
        self.numLabels = y.shape[1]
        for i in range(self.numModels):
            print("Model-", i+1, "Sampling")
            mlSamplingCopy = copy.deepcopy(self.mlSampling)
            mlSamplingCopy.setRandomState(i+self.randomState)
            Xnew, ynew = mlSamplingCopy.fit_resample(X, y)
            model = copy.deepcopy(self.baseMultiLabelLearner)
            model.fit(Xnew, ynew)
            self.mlls.append(model)
        print("Calculating thresholds")
        self.calculateThresholds(X, y)

    def predict(self, X):
        nData = len(X)
        result = []
        for i in range(nData):
            bipartition, confidences = self.makePredictionSingleData(X[i])
            result.append(bipartition)
        return np.asarray(result)

    def predict_proba(self, X):
        nData = len(X)
        result = []
        for i in range(nData):
            bipartition, confidences = self.makePredictionSingleData(X[i])
            result.append(confidences)
        return np.asarray(result)

    def makePredictionSingleData(self, x1):
        conf = [0]*self.numLabels
        for i in range(self.numModels):
            confidences = self.mlls[i].predict_proba([x1])[0]
            for j in range(self.numLabels):
                conf[j]+=confidences[j]

        for j in range(self.numLabels):
            conf[j] /= self.numModels

        bipartition = []
        for j in range(self.numLabels):
            bipartition.append(conf[j] >= self.thresholds[j])

        return bipartition, conf

    def calculateThresholds(self, X, y):
        self.thresholds = []
        numInstances = y.shape[0]
        numLabels = y.shape[1]
    	# thresholdOptimizationMeasures m=measure;
    	# measure=thresholdOptimizationMeasures.None;
        
        predictConfidences = []#new double [trainingSet.getNumInstances()][trainingSet.getNumLabels()];
        for i in range(numInstances):
            Xdata, Ydata = X[i], y[i]
            bipartition, conf = self.makePredictionSingleData(Xdata)
            predictConfidences.append(conf)

        for j in range(numLabels):
            maxVal = -1000000000000.0
            trueLabels = [ data[j]==1 for data in y]
            
            d = 0.05
            while d<1:
                predictLabels = [predictConfidences[i][j]>=d for i in list(range(numInstances))]
                #Using Fmeasure
                value = f1_score(trueLabels, predictLabels, average='macro')
                if value > maxVal:
                    maxVal = value
                    self.thresholds[j] = d
                d+=0.05

    
