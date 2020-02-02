from skmultilearn.dataset import load_dataset
# from BinaryRelevanceUnderSampling import BinaryRelevanceUnderSampling
from CocoaXGBoost import CocoaXGBoost
clf = CocoaXGBoost()
X_train, y_train, feature_names, label_names = load_dataset('emotions', 'train')
X_test, y_test, _, _ = load_dataset('emotions', 'test')


# print(type(y_train))
# print(y_train)
# print(type(X_train))
# print(X_train)
# print(type(y_train.toarray()[0]))
# from skmultilearn.problem_transform import BinaryRelevance
# from BinaryRelevanceUnderSampling import BinaryRelevanceUnderSampling
# from sklearn.svm import SVC


# clf = BinaryRelevanceUnderSampling(
#     classifier=SVC(),
#     require_dense=[False, True]
# )
# clf.fit(X_train, y_train)