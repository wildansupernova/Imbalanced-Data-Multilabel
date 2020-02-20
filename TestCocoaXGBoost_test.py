import unittest
from CocoaXGBoost import CocoaXGBoostUndersampling
import numpy as np
from xgboost import XGBClassifier
from scipy import sparse

class mockBrus:
    def predict_proba(self, X):
        return sparse.csr_matrix([[0.5, 0.5, 0.5]])

class mockTriClassifier:
    def predict_proba(self, X):
        return np.asarray([[0.4, 0.4, 0.2]])

class TestCocoaXGBoost_test(unittest.TestCase):

    def test_generate_partition(self):
        print("====test_generate_partition")
        obj = CocoaXGBoostUndersampling()
        X = np.asarray([
            [1,1,0,1],
            [0,1,0,1],
            [0,1,1,1],
            [0,1,1,1],
            [0,1,1,1],
            [0,1,1,1],
            [0,1,1,1],
            [0,1,1,1],
            [0,1,1,1]
        ])
        y = np.asarray([
            [0,1,1],
            [0,1,1],
            [0,1,1],
            [0,1,1],
            [0,1,1],
            [0,1,1],
            [0,1,1],
            [0,1,1],
            [0,1,1],
        ])
        obj._generate_partition(X, y)
        self.assertTrue(obj.partition_,[0,1,2])
        self.assertTrue(obj.labelIndices,[0,1,2])
        self.assertEqual(obj.model_count_,3)
        self.assertEqual(obj._label_count,3)

    def test_TrirandomUnderSampling(self):
        print("====test_TrirandomUnderSampling")
        obj = CocoaXGBoostUndersampling()
        X = np.asarray([
            [1,1,0,1],
            [0,1,0,1],
            [0,1,1,1],
            [0,1,1,1],
            [0,1,1,1],
            [0,1,1,1],
            [0,1,1,1],
            [0,1,1,1],
            [0,1,1,1]
        ])
        y = np.asarray([
            1,
            2,
            1,
            2,
            2,
            3,
            3,
            3,
            3,
        ])
        result = obj.TrirandomUnderSampling(X, y)
        unique_elements, counts_elements = np.unique(y, return_counts=True)
        dictCount = dict(zip(unique_elements, counts_elements))
        self.assertEqual(dictCount[1],2)
        self.assertEqual(dictCount[2],3)
        self.assertEqual(dictCount[3],4)
    
    def test_selectedLabelIndices(self):
        print("====test_selectedLabelIndices")
        obj = CocoaXGBoostUndersampling()
        labelIndexList = [1,2,3,4,5]
        currentLabelIndex = 3
        obj.numCouples = 4
        result = obj.selectedLabelIndices(labelIndexList, currentLabelIndex)
        self.assertTrue(result == [1,2,4,5])

    def getObjTest(self):
        X = np.asarray([
            [1,1,0,1],
            [0,1,0,1],
            [0,1,1,1],
            [0,1,1,1],
            [0,1,1,1],
            [0,1,1,1],
            [0,1,1,1],
            [0,1,1,1],
            [0,1,1,1]
        ])
        y = np.asarray([
            [0,1,0],
            [1,0,1],
            [0,1,1],
            [0,1,1],
            [0,1,1],
            [0,1,1],
            [0,1,1],
            [0,1,1],
            [0,1,1],
        ])
        obj = CocoaXGBoostUndersampling()
        obj._label_count = 3
        obj.numCouples = 2
        obj.brus = mockBrus()
        obj.triClassifiers = [[mockTriClassifier()]*2]*3
        obj.thresholds = [0.4, 0.1, 0.35]
        return obj

    def test_makePredictionforThreshold(self):
        obj = self.getObjTest()
        self.assertTrue(obj.makePredictionforThreshold([1,1,0,1]) == [0.3, 0.3, 0.3])
        return obj
    

    def test_makePredictionSingleData(self):
        obj = self.getObjTest()
        bipartition, confidences = obj.makePredictionSingleData([1,1,0,1])
        self.assertTrue( bipartition == [0, 1, 0])

    def test_predict(self):
        obj = self.getObjTest()
        bipartitions = obj.predict([[1], [2]])
        self.assertTrue( (bipartitions[0] == [0, 1, 0]).all())
        self.assertTrue( (bipartitions[1] == [0, 1, 0]).all())
    
    def test_predict_proba(self):
        obj = self.getObjTest()
        bipartitions = obj.predict_proba([[1], [2]])
        self.assertTrue( (bipartitions[0] == [0.3, 0.3, 0.3]).all())
        self.assertTrue( (bipartitions[1] == [0.3, 0.3, 0.3]).all())

    def test_calculateThresholds(self):
        print("====test_TrirandomUnderSampling")
        X = np.asarray([
            [1,1,0,1],
            [0,1,0,1],
            [0,1,1,1],
            [0,1,1,1],
            [0,1,1,1],
            [0,1,1,1],
            [0,1,1,1],
            [0,1,1,1],
            [0,1,1,1]
        ])
        y = np.asarray([
            [0,1,0],
            [1,0,1],
            [0,1,1],
            [0,1,1],
            [0,1,1],
            [0,1,1],
            [0,1,1],
            [0,1,1],
            [0,1,1],
        ])
        obj = self.getObjTest()
        obj.calculateThresholds(X, y)
        self.assertTrue(obj.thresholds == [0.35, 0.05, 0.05])

