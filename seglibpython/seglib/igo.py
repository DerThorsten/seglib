import numpy

def ToyObjective(objective):
	def __init__(self):
		pass


	def numVar(objective):
		pass

	def evaluate(objective):
		pass


class Igo(self):
	def __init__(self,objective):
		self.objective 	= objective
		self.numVar_	= objective.numVar()
		self.mean 	   	= numpy.zeros(self.numVar_)
		self.covariance = numpy.ones([self.numVar_,self.numVar_])

	def setStartingPoint(self,mean,covariance):
		self.mean[:] 		= mean[:]
		self.covariance[:]  = covariance[:]



	def generateSamples(self,n):
		return  numpy.random.multivariate_normal(self.mean, self.covariance, n)


	def evaluateSamples(self,samples):
		""" get the value of the objective for each sample"""
		assert samples.ndim==2 :
		assert samples.shape[0] == self.numVar

		objectiveValue = numpy.zeros(samples.shape[1])
		for si in range(samples.shape[0])
			sample = samples[:,si]
			objectiveValue[i]=self.objective.evaluate(sample)

		return objectiveValue

	def rankSamples(self,objectiveValues,selectionQuantile=0.27):
		numberOfSamples = len(objectiveValues)
		sampleRank 		= numpy.argsort(objectiveValues).astype(numpy.float32)

		# argument of quantile selction function
		w	= (sampleRank +0.5)/float(numberOfSamples)
		# find where the quantile is to large
		whereQ=numpy.where(w>selectionQuantile)
		w[:]=1.0
		w[whereQ]=0.0
		w/=float(numberOfSamples)

