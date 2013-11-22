import numpy

import scipy.spatial.distance
from seglib.histogram import histDist,moveMe



def genericDist(A,B,distance):
    A.astype(numpy.float64)
    B.astype(numpy.float64)
    out = scipy.spatial.distance.cdist(A,B,distance)
    return out

def histDiff(A,B,distance):

    out = numpy.zeros([A.shape[0],B.shape[0]],dtype=numpy.float32)

    #def printInf(a,name):
        #print name,a.shape,a.dtype

    #printInf(A,"A")
    #printInf(B,"B")
    #printInf(out,"out")

    histDist(A,B,out)
    return out

class MiniBatchKMeans(object):

    def __init__( self, nClusters=8, init='random',nInit=10,batchSize=100,iterations=100,verbose=False,shape=None,distance="euclidean"):

        # parameters
        self.nClusters  = nClusters
        self.init       = init
        self.nInit      = nInit
        self.batchSize  = batchSize
        self.verbose    = verbose
        self.iterations = iterations
        self.distance   = distance
        # fit results
        self._labels    = None
        self._centers   = None
        self._bestCenters  = None
        self._bestE        = None

        # working data
        self._randomIndices = None
        self._nSamples      = None
        self._nFeatures     = None
        self._X             = None
        self._IBatch        = None
        self._XBatch        = None
        self._centerCount   = numpy.zeros(self.nClusters,dtype=numpy.float32)

        if distance in ["bhattacharyya","chi2"]:
            self.distComputer   = histDiff
        else:
            self.distComputer   = genericDist

    def fit(self,X):
        self._bestCenters  = None
        self._bestE        = None

        self._X = X
        self._nSamples,self._nFeatures = self._X.shape
        self._randomIndices = numpy.arange(self._nSamples,dtype=numpy.int64) 
        self._IBatch        = numpy.random.permutation(self._randomIndices)[:self.batchSize]
        self._XBatch        = self._X[self._IBatch[:],:]
        # initialize centers
        self._getStartingCenters() # will initals self.centers_

        for ii in range(1):
            print "reset"
            self._centerCount[:]=0.0
            for i in range(self.iterations):
                #print "i",i

                self._IBatch      = numpy.random.permutation(self._randomIndices)[:self.batchSize]
                self._XBatch      = self._X[self._IBatch[:],:]
                minDistCluster ,e =  self.minDistCenter(centers=self._centers,features=self._XBatch)
                
                """
                def printInf(a,name):
                    print name,a.shape,a.dtype

                printInf(self._X,"X")
                printInf(self._IBatch,"_IBatch")
                printInf(minDistCluster,"minDistCluster")
                printInf(self._centerCount,"_centerCount")
                printInf(self._centers,"_centers")
                """                
                
                # this functions needs to be refactored   
                #"""
                moveMe(
                    globalFeatures=self._X,
                    batchIndex=self._IBatch,
                    minCenterIndex=minDistCluster,
                    centerCount=self._centerCount,
                    centers=self._centers
                )
                """
                self._XBatch    = self._X[self._IBatch[:],:]
                for bb in range(self.batchSize):
                    c = minDistCluster[bb]
                    self._centerCount[c]+=1.0

                    n = 1.0/(self._centerCount[c])

                    #centers[c,:]   = (1.0-n)*centers[c,:] + n*batchFeatures[bb,:]

                    self._centers[c,:]*=(1.0-n)
                    self._centers[c,:]+=n*self._XBatch[bb,:]
                """

                if i%60 == 0:
                    labels , e = self.minDistCenter(self._centers,self._X)
                    print i,self.iterations,e
                #print "centers"
                #print centers

        self._labels,e,distCluster=self.minDistCenter(self._centers,self._X,returnDistMatrix=True)


        self._distToCenters = distCluster


    def _getStartingCenters(self):
        testCenters = numpy.zeros([self.nInit,self.nClusters, self._nFeatures],dtype=numpy.float32)
        testErrors  = numpy.zeros([self.nInit])
        for i in range(self.nInit):
            randSampleIndex  = numpy.random.permutation(self._randomIndices)[:self.nClusters] 
            testCenters[i,:,:] = self._X[randSampleIndex,:]
            labes,e = self.minDistCenter(testCenters[i,:,:] ,self._X)
            testErrors[i] = e

            print testErrors[i] 

        bestTestCenterIndex = numpy.argmin(testErrors)
        self._centers=testCenters[bestTestCenterIndex,:]
        

    def minDistCenter(self,centers,features,returnDistMatrix=False):
        nCenters,nFeatures = centers.shape[:2]
        nSamples        = features.shape[0]
        distance        = self.distComputer(features,centers,self.distance)
        minDistCluster  = numpy.argmin(distance,axis=1)
        minDist         = numpy.min(distance,axis=1)
        if returnDistMatrix==False:
            return minDistCluster,numpy.sum(minDist)
        else:
            return minDistCluster,numpy.sum(minDist),distance










