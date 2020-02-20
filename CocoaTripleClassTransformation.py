import numpy as np
class CocoaTripleClassTransformation:
    def __init__(self, y):
        """
        Constructor
        =====
        y : `array_like`, shape -> nx1
        """
        self._y = y
        self._label_count = y.shape[1]
        self.numInstances = y.shape[0]

    def transformLabelUtil(self, yi, idxLabel1, idxLabel2):
        label1 = yi[idxLabel1]
        label2 = yi[idxLabel2]
        if label1 == 0:
            if label2 == 0:
                return 0
            elif label2 == 1:
                return 1
        elif label1 == 1:
            return 2

    def transformLabels(self, idxLabel1, idxLabel2):
        result = []
        for i in range(self.numInstances):
            temp = self.transformLabelUtil(self._y[i], idxLabel1, idxLabel2)
            result.append(temp)
        return np.asarray(result)

"""
How to test ?
1. Test transformLabelUtil according to Equation 4 at Zhang Paper
2. transformLabels with more than one data
"""