# from imblearn.over_sampling.base import BaseOverSampler
import random
from sklearn.neighbors import KNeighborsClassifier

class MLSOL():
    def __init__(self, numOfNeighbors = 5, ratio = 0.1, randomState = 1):
        self._weights = []
        self._C = [] # C[][] the C_ij for majority class is null
        self._insTypes = [] # insTypes[][]
        self._knnIndices = [] # type [][]
        self._minLabels = []
        self._labelIndices = []
        self._featureIndices = []
        self._sumW = None
        self._numOfNeighbors = numOfNeighbors
        self._percentageGeneratedInstance = ratio
        self._randomState = randomState

    def countC1C0(self, y, numLabels):
        c1 = [0]*numLabels
        c0 = [0]*numLabels

        for e in y:
            for j in range(numLabels):
                if e[j] == 0:
                    c0[j]+=1
                elif e[j] == 1:
                    c1[j]+=1

        return c0, c1

    def getMinLabels(self, y, numLabels):
        c0, c1 = self.countC1C0(y, numLabels)
        minLabels = []

        for i in range(numLabels):
            minLabels.append(1 if c1[i] > c0[i] else 0)
        return minLabels

    def fit_resample(self, X, y):
        rnd = random.Random(self._randomState)
        numLabels = y[1]
        self._labelIndices = list(range(numLabels))
        # featureIndices=mlDataset.getFeatureIndices();
        nData = len(y)
        generatedNumberIns = int(nData * self._percentageGeneratedInstance)
        self.knnClassifier = KNeighborsClassifier(n_neighbors=self._numOfNeighbors)

		# weights=new double[oriNumIns];
        self.minLabels = self.getMinLabels(y, numLabels)
		
		# C=new Double[oriNumIns][numLabels];
		self._knnIndices = [] # knnIndices=new int[nData][numOfNeighbors];

		# Instances ins=mlDataset.getDataSet();
		# calculateWeight(ins);
		# initilizeInsTypes(ins);
        xNew = X.copy()
        yNew = y.copy()

		# for(int i=0;i<generatingNumIns;i++){
		# 	double d=rnd.nextDouble()*sumW;
		# 	int centralIndex=-1;
		# 	double s=0;
		# 	for(int j=0;j<weights.length;j++){
		# 		s+=weights[j];
		# 		if(d<=s){
		# 			centralIndex=j;
		# 			break;
		# 		}
		# 	}
		# 	int referenceIndex=knnIndices[centralIndex][rnd.nextInt(numOfNeighbors)];
		# 	Instance newData=generateSyntheticInstance(ins.get(centralIndex), ins.get(referenceIndex), centralIndex, referenceIndex, rnd);
		# 	insNew.add(newData);
		# }
		
		
		# return new MultiLabelInstances(insNew, mlDataset.getLabelsMetaData());


    def _calculate_weight(self, X, y): 
        numInstances = len(y)
        numLabels = len(self._labelIndices)
        self._knnClassifer = KNeighborsClassifier(n_neighbors=self._numOfNeighbors).fit(X, y).f
        self._knnIndices = [] # knnIndices=new int[nData][numOfNeighbors];
        for i in range(numInstances):
            xData = X[i]
            yData = y[i]
            knnLabel = self._knnClassifer





    def _initilizeIns_types(self): pass

    def _generate_synthetic_instance(self): pass