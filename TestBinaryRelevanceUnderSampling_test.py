import unittest
from BinaryRelevanceUnderSampling import BinaryRelevanceUnderSampling
import numpy as np
from xgboost import XGBClassifier

class TestBinaryRelevanceUndersampling(unittest.TestCase):
    
    def test_BR(self):
        print("====TestBinaryRelevanceUndersampling")
        obj = BinaryRelevanceUnderSampling(classifier=XGBClassifier())
        obj.fit(np.asarray([[1,2,3], [4,5,6]]), np.asarray([[1,0],[0,1]]))
        self.assertEqual(2, len(obj.classifiers_))
        self.assertEqual(2, obj._label_count)
        