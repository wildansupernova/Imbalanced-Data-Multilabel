import unittest
from MLSOL import MLSOL
import numpy as np

class TestMLSOL_test(unittest.TestCase):
    
    def test_countC1C0(self):
        y = [
            [1,1,0,0],
            [1,0,1,1],
            [1,0,0,1]
        ]
        numLabels = 4

        obj = MLSOL()
        c0, c1 = obj.countC1C0(y,numLabels)
        self.assertTrue(c0 == [0,2,2,1])
        self.assertTrue(c1 == [3,1,1,2])

    def test_getMinLabels(self):
        y = [
            [1,1,0,0],
            [1,0,1,1],
            [1,0,0,1]
        ]
        numLabels = 4

        obj = MLSOL()
        minLabels= obj.getMinLabels(y,numLabels)
        self.assertTrue(minLabels == [1,0,0,1])