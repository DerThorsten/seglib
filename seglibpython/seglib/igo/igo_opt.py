import numpy

def ToyObjective(objective):
    def __init__(self):
        pass


    def numVar(objective):
        pass

    def evaluate(objective):
        pass


class Igo(object):
    def __init__(self,objective):
        self.objective  = objective
        self.numVar_    = objective.numVar()
        self.mean       = numpy.zeros(self.numVar_)
        self.covariance = numpy.eye(self.numVar_)

    def setStartingPoint(self,mean,covariance):
        self.mean[:]        = mean[:]
        self.covariance[:]  = covariance[:]



    def generateSamples(self,n):
        return  numpy.random.multivariate_normal(self.mean, self.covariance, n)


    def evaluateSamples(self,samples):
        """ get the value of the objective for each sample"""
        assert samples.ndim==2 
        print samples.shape
        assert samples.shape[1] == self.numVar_

        objectiveValue = numpy.zeros(samples.shape[0])
        for si in range(samples.shape[0]):
            sample = samples[si,:]
            objectiveValue[si]=self.objective.evaluate(sample)

        return objectiveValue

    def rankSamples(self,objectiveValues,selectionQuantile=0.027):
        numberOfSamples = len(objectiveValues)
        sampleRank      = numpy.argsort(objectiveValues).astype(numpy.float32)

        # argument of quantile selction function
        w   = (sampleRank +0.5)/float(numberOfSamples)
        # find where the quantile is to large
        whereQ=numpy.where(w>selectionQuantile)
        w[:]=1.0
        w[whereQ]=0.0
        w/=float(numberOfSamples)
        return w

    def update(self,samples,w,etaMean,etaCovariance):
        numberOfSamples = samples.shape[0]
        newMean         = numpy.zeros(self.mean.shape)
        newCovariance   = numpy.zeros(self.covariance.shape)

        print "wshape ",w.shape
        print "s",      samples.shape

        for s in range(numberOfSamples):
            d   = samples[s]-self.mean
            dM  = d.reshape([-1,1])
            dMT = dM.T
            newMean         +=w[s]*d
            dotRes = numpy.dot(dM,dMT)
            #print "dosRes ",dotRes.shape
            #print "cov    ",self.covariance.shape
            newCovariance   +=w[s]*(dotRes-self.covariance)

        # gradient step
        self.mean       += etaMean*newMean
        self.covariance += etaCovariance*newCovariance

    def infer(self):
        
        for i in range(1000):
            print i
            samples = self.generateSamples(self.numVar_*20)
            print "eval ",samples.shape
            objectiveValues  = self.evaluateSamples(samples)
            w                = self.rankSamples(objectiveValues=objectiveValues)
            if i%100==0:
                self.objective.show(samples[numpy.argmin(w),:])
            print "min objectiveValues",objectiveValues.min()
            self.update(samples=samples,w=w,etaMean=1.0,etaCovariance=1.0)

