import unittest
from CocoaTripleClassTransformation import CocoaTripleClassTransformation
import numpy as np

class TestCocoaTripleClassTransformation(unittest.TestCase):
    
    def test_transformLabelUtil(self):
        print("===TestCocoaTripleClassTransformation - test_transformLabelUtil")
        y = np.asarray([
            [1,1,0,1],
            [0,1,0,1],
            [0,1,1,1]
        ])
        obj = CocoaTripleClassTransformation(y)
        self.assertEqual(0, obj.transformLabelUtil(y[1], 0, 2)) 
        self.assertEqual(1, obj.transformLabelUtil(y[1], 0, 1)) 
        self.assertEqual(2, obj.transformLabelUtil(y[1], 1, 2)) 
    
    def test_transformLabels(self):
        print("===TestCocoaTripleClassTransformation - test_transformLabels")
        y = np.asarray([
            [1,1,0,1],
            [0,1,0,1],
            [0,0,1,1]
        ])
        obj = CocoaTripleClassTransformation(y)
        self.assertTrue((np.asarray([2,1,0]) == obj.transformLabels(0,1)).all())
