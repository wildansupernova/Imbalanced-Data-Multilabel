# from imblearn.over_sampling.base import BaseOverSampler
import random
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
class InstanceType:
    SAFE =  0
    BORDERLINE = 1
    RARE = 2
    OUTLIER = 3
    MAJORITY = 4
    insTypeTheta = {
        0:0.5,
        1:0.75,
        2:1.0+1e-5,
        3:0.0-1e-5,
    }

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

    def setRatio(self, p):
        self._percentageGeneratedInstance = p
    
    def getRatio(self):
        return self._percentageGeneratedInstance

    def setRandomState(self, randomState):
        self._randomState = randomState
    
    def getRandomState(self):
        return self._randomState

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
        numLabels = y.shape[1]
        self._labelIndices = list(range(numLabels))
        self._featureIndices = list(range(X.shape[1]))
        nData = len(y)
        generatedNumberIns = int(nData * self._percentageGeneratedInstance)
        self.knnClassifier = KNeighborsClassifier(n_neighbors=self._numOfNeighbors)

		# weights=new double[oriNumIns];
        self._minLabels = self.getMinLabels(y, numLabels)
		
		

        self._calculate_weight(X, y)
        self._initilizeIns_types(X, y)
        xNew = X.copy()
        yNew = y.copy()
        xNewAdd = []
        yNewAdd = []
        for  i in range(generatedNumberIns):
            d = rnd.uniform(0, 1)
            centralIndex = -1
            s = 0
            for j in range(nData):
                s+= self._weights[j]
                if d<=s:
                    centralIndex = j
                    break
            referenceIndex = self._knnIndices[centralIndex][rnd.randint(0, self._numOfNeighbors - 1)]
            # Instance newData=generateSyntheticInstance(ins.get(centralIndex), ins.get(referenceIndex), centralIndex, referenceIndex, rnd);
            xNewAddTemp, yNewAddTemp = self._generate_synthetic_instance(X[centralIndex],y[centralIndex], X[referenceIndex], y[referenceIndex], centralIndex, referenceIndex, rnd)
            xNewAdd.append(xNewAddTemp)
            yNewAdd.append(yNewAddTemp)

        return np.concatenate((xNew,xNewAdd)), np.concatenate((yNew,yNewAdd)) # return new MultiLabelInstances(insNew, mlDataset.getLabelsMetaData()); X,y
    
    def _calculate_weight(self, X, y): 
        numInstances = len(y)
        numLabels = len(self._labelIndices)
        self._knnClassifer = KNeighborsClassifier(n_neighbors=self._numOfNeighbors).fit(X, y)
        self._knnIndices = [] # knnIndices=new int[nData][numOfNeighbors];
        self._C = [] # C=new Double[oriNumIns][numLabels];
        for i in range(numInstances):
            self._C.append([])
            xData = X[i]
            yData = y[i]
            result = self._knnClassifer.kneighbors([xData], return_distance=False)[0]
            for j in range(self._numOfNeighbors):
                self._knnIndices.append(result)

            for j in range(numLabels):
                numMaj = 0
                if yData[self._labelIndices[j]] == self._minLabels[j]:
                    for k in range(self._numOfNeighbors):
                        if yData[self._labelIndices[j]] != y[result[k]][self._labelIndices[j]]:
                            numMaj+=1
                    self._C[i].append(numMaj*1.0/self._numOfNeighbors)
                else:
                    self._C[i].append(None)


        # //Transform the C to scores
        scores = [ [0.0]*numLabels for e in range(numInstances)] # Double scores[][]=new Double[numIns][numLabels];

        for j in range(numLabels):
            sum=0.0
            c=0
            for i in range(numInstances):
                if self._C[i][j] != None and self._C[i][j] < 1 :
                    sum+=self._C[i][j]
                    c+=1
            if c!=0 and sum != 0.0:
                for i in range(numInstances):
                    if self._C[i][j] !=None and self._C[i][j] < 1:
                        scores[i][j] =  self._C[i][j]/sum
        
        self._sumW=0.0
        self._weights = []
        for i in range(numInstances):
            self._weights.append(0.0)
            for j in range(numLabels):
                if scores[i][j] != None:
                    self._weights[i] += scores[i][j]
            self._sumW+=self._weights[i]

    def _initilizeIns_types(self, X, y):
        numInstances = len(y)
        numLabels = len(self._labelIndices)
        self._insTypes = [] # new InstanceType[numIns][labelIndices.length];
        for i in range(numInstances):
            self._insTypes.append([])
            yData = y[i]
            for j in range(numLabels):
                if yData[self._labelIndices[j]] == self._minLabels[j]:
                    if self._C[i][j] < 0.3 :
                        self._insTypes[i].append(InstanceType.SAFE)
                    elif self._C[i][j] < 0.7:
                        self._insTypes[i].append(InstanceType.BORDERLINE)
                    elif self._C[i][j] < 1:
                        self._insTypes[i].append(InstanceType.RARE)
                    else:
                        self._insTypes[i].append(InstanceType.OUTLIER)
                else:
                    self._insTypes[i].append(InstanceType.MAJORITY)

		
		# //re-analyse the RARE type
        flag = True
        while flag:
            flag = False
            for i in range(numInstances):
                for j in range(numLabels):
                    if self._insTypes[i][j] == InstanceType.RARE:
                        for k in self._knnIndices[i]:
                            if self._insTypes[k][j]==InstanceType.SAFE or self._insTypes[i][j]==InstanceType.BORDERLINE:
                                self._insTypes[i][j]=InstanceType.BORDERLINE
                                flag = True
                                break

    def _generate_synthetic_instance(self,XCentralInstance, YCentralInstance, XReferenceInstance, YReferenceInstance, centralIndex, referenceIndex, rnd):
        numFeatures = len(self._featureIndices)
        numLabels = len(self._labelIndices)
        xNew = XCentralInstance.copy()
        yNew = YCentralInstance.copy()
        for i in range(numFeatures):
            xNew[i] += rnd.uniform(0, 1)*(XReferenceInstance[i]- XCentralInstance[i])
        d1 = np.linalg.norm(XCentralInstance - xNew)
        d2 = np.linalg.norm(XReferenceInstance - xNew)
        cd = 0.5 if d1 == 0 and d2 == 0  else (d1/(d1+d2))
        theta = 0.5

        for i in range(numLabels):
            j = self._labelIndices[i]
            if YCentralInstance[j] == YReferenceInstance[j]:
                yNew[j] = YCentralInstance[j]
            else:
                if self._insTypes[centralIndex][i] == InstanceType.MAJORITY:
                    temp = (XCentralInstance, YCentralInstance)
                    XCentralInstance, YCentralInstance =  (XReferenceInstance, YReferenceInstance)
                    XReferenceInstance, YReferenceInstance = temp
                    temp = centralIndex
                    centralIndex = referenceIndex
                    referenceIndex = temp
                    cd = 1.0 - cd
                theta = InstanceType.insTypeTheta[self._insTypes[centralIndex][i]]
                if cd <= theta:
                    yNew[j] = YCentralInstance[j]
                else:
                    yNew[j] = YReferenceInstance[j]
        return xNew, yNew