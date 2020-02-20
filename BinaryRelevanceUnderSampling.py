import copy
import numpy as np

from scipy.sparse import hstack, issparse, lil_matrix
from skmultilearn.problem_transform import BinaryRelevance
from imblearn.under_sampling import RandomUnderSampler

class BinaryRelevanceUnderSampling(BinaryRelevance):
    def __init__(self, classifier=None, require_dense=None, seed=1):
        super(BinaryRelevanceUnderSampling, self).__init__(
            classifier, require_dense)
        self.seed = seed

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
        X = self._ensure_input_format(
            X, sparse_format='csr', enforce_sparse=True)
        y = self._ensure_output_format(
            y, sparse_format='csc', enforce_sparse=True)

        self.classifiers_ = []
        self._generate_partition(X, y)
        self._label_count = y.shape[1]

        rus = RandomUnderSampler(random_state=self.seed)

        for i in range(self.model_count_):
            classifier = copy.deepcopy(self.classifier)
            y_subset = self._generate_data_subset(
                y, self.partition_[i], axis=1)

            X_resampled, y_resampled = rus.fit_resample(X,y_subset.toarray())
            
            if issparse(y_resampled) and y_resampled.ndim > 1 and y_resampled.shape[1] == 1:
                y_resampled = np.ravel(y_resampled.toarray())
            classifier.fit(self._ensure_input_format(
                X_resampled), self._ensure_output_format(y_resampled))
            self.classifiers_.append(classifier)
        return self


"""
How to test?
1. Check label count if it's the same of X,y
2. Check classifier number, it should give the same with label count
"""