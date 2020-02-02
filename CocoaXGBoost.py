from skmultilearn.base import ProblemTransformationBase
from BinaryRelevanceUnderSampling import BinaryRelevanceUnderSampling
from CocoaTripleClassTransformation import CocoaTripleClassTransformation
from imblearn.datasets import make_imbalance
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import numpy as np
import random
import copy

paramXGBoost = {
    "eta":0.2,
    "gamma":0,
    "min_child_weight": 1,
    "max_depth": 3,
    "colsample_bytree": 0.7
}

paramXGBoostMulticlass = {
    "eta":0.2,
    "gamma":0,
    "min_child_weight": 1,
    "max_depth": 3,
    "colsample_bytree": 0.7,
    "objective": "multi:softmax"
}


class CocoaXGBoost(ProblemTransformationBase):
    def __init__(self, numMaxCouples = 10, underSamplingPercent = 1.0, seed = 1):
        super(CocoaXGBoost, self).__init__(XGBClassifier(**paramXGBoost), None)
        self.multiclassClassifier = XGBClassifier(**paramXGBoostMulticlass)
        self.numMaxCouples = numMaxCouples
        self.underSamplingPercent = underSamplingPercent
        self.seed = seed
        self.numCouples = None

    def getNumMaxCouples(self):
        return self.numMaxCouples
    
    def getNumCouples(self):
        return self.numCouples
    
    def getUnderSamplingPercent(self):
        return self.underSamplingPercent

    def setUnderSamplingPercent(self, underSamplingPercent):
        self.underSamplingPercent = underSamplingPercent
    
    def getSeed(self):
        return self.seed

    def setSeed(self, seed):
        self.seed = seed

    def _generate_partition(self, X, y):
        """Partitions the label space into singletons
        Sets `self.partition_` (list of single item lists) and `self.model_count_` (equal to number of labels).
        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            not used, only for API compatibility
        y : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix of `int`, shape=(n_samples, n_labels)
            binary indicator matrix with label assignments
        """
        self.partition_ = list(range(y.shape[1]))
        self.labelIndices = list(range(y.shape[1]))
        self.model_count_ = y.shape[1]
        self._label_count = y.shape[1]

    def fit(self, X, y):
        """Fits classifier to training data
        Parameters
        ----------
        X : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix, shape=(n_samples, n_features)
            input feature matrix
        y : `array_like`, :class:`numpy.matrix` or :mod:`scipy.sparse` matrix of `{0, 1}`, shape=(n_samples, n_labels)
            binary indicator matrix with label assignments
        Returns
        -------
        self
            fitted instance of self
        Notes
        -----
        .. note :: Input matrices are converted to sparse format internally if a numpy representation is passed
        """
        self._generate_partition(X, y)
        self.numCouples = min(self.getNumMaxCouples(), self._label_count-1)
        self.brus = BinaryRelevanceUnderSampling(self.classifier, seed=self.seed)
        self.trt = CocoaTripleClassTransformation(y)
        self.triLabelIndices = []
        self.triClassifiers = []
        for i in range(self._label_count):
            self.triLabelIndices.append([]) # Actually init indices
            self.triClassifiers.append([]) # Actually init classifier
            for j in range(self.numCouples):
                self.triClassifiers[i].append(copy.deepcopy(self.multiclassClassifier)) # AbstractClassifier.makeCopy(baseClassifier);	
        self.thresholds = [-1]*self._label_count #Init threshold list
        
        self.brus.fit(X, y)
		
        labelIndexList = []
        for i in range(self._label_count):
            labelIndexList.append(self.labelIndices[i])

        rnd = random.Random(self.seed)
        for i in range(self._label_count):
            rnd.shuffle(labelIndexList)
            self.triLabelIndices[i] = self.selectedLabelIndices(labelIndexList, self.labelIndices[i])
            for j in range(self.numCouples):
                yTriClassIns = self.trt.transformLabels(self.labelIndices[i], self.triLabelIndices[i][j])
                xUsTriClassIns, YUsTriClassIns = self.TrirandomUnderSampling(X, yTriClassIns)
                self.triClassifiers[i][j].fit(xUsTriClassIns, YUsTriClassIns)
       	self.calculateThresholds(X, y)

    def selectedLabelIndices(self, labelIndexList, currentLabelIndex):
        result = []
        i_list = 0
        i_array = 0
        while i_array<self.numCouples:
            l=labelIndexList[i_list]
            if l!=currentLabelIndex:
                result.append(l)
                i_array+=1
            i_list+=1
        return result

    def TrirandomUnderSampling(self, X, y): 
        """
        y : numpy array
        """
        result = []
        unique_elements, counts_elements = np.unique(y, return_counts=True)
        numClass = len(unique_elements)
        c = [0]*numClass
        nData = len(y)
        minVal = counts_elements.min()

        sample_strategy = dict()
        for i in range(numClass):
            sample_strategy[i] = minVal
        print(y)
        Xres, yres = make_imbalance(X,y,sample_strategy)
        return Xres, yres

    def makePredictionforThreshold(self, xData):
        confidences = [0]*self._label_count
        X = np.asarray([xData])
        yPredProba = self.brus.predict_proba(X).toarray()[0]

        for i in range(self._label_count):
            confidences+=yPredProba[i]

        for j in range(self._label_count):
            for k in range(self.numCouples):
                d = self.triClassifiers[j][k].predict_proba(np.asarray([xData]))
                confidences[j] += d[0][2]
            confidences[j] /= (self.numCouples+1)
        return confidences

    def calculateThresholds(self, X, y):
        nData = y.shape[0]
        nLabel = y.shape[1]
        predictConfidences = []
        for i in range(nData):
            predictConfidences.append(self.makePredictionforThreshold(X[i]))

        for j in range(self._label_count):
            maxVal = -1000000000000.0
            trueLabels = [ data[j]==1 for data in y]
            
            d = 0.05

            while d<1:
                predictLabels = [predictConfidences[i][j]>=d for i in list(range(nData))]
                #Using Fmeasure
                value = f1_score(trueLabels, predictLabels, average='macro')
                if value > maxVal:
                    maxVal = value
                    self.thresholds[j] = d
                d+=0.05 

    def predict(self, X):
        nData = len(X)
        result = []
        for i in range(nData):
            bipartition, confidences = self.makePredictionSingleData(X[i])
            result.append(bipartition)
        return np.asarray(result)

    def makePredictionSingleData(self, x1):
        confidences = self.makePredictionforThreshold(x1)
        bipartition = [0]*self._label_count
        for j in range(self._label_count):
            bipartition[j] = int(confidences[j] > self.thresholds[j])

        return bipartition, confidences
