import vigra
import numpy
import opengm
from seglib import cgp2d 
from seglib.clustering.ce_multicut import *
from seglib.preprocessing import norm01,normCProb,reshapeToImage,normCProbFlat
from sklearn.cluster import Ward,WardAgglomeration
import pylab
import sys
from optparse import OptionParser
import inspect



img = "img/37073.jpg"
#img = "img/156065.jpg"
img="img/42049.jpg"
img="img/zebra.jpg"
img = numpy.squeeze(vigra.readImage(img))#[0:75,0:75,:]
lab = vigra.colors.transform_RGB2Lab(img)
labels ,nseg= vigra.analysis.slicSuperpixels(lab,10.0,3)
labels = vigra.analysis.labelImage(labels).astype(numpy.uint64)
cgp,tgrid = cgp2d.cgpFromLabels(labels)
imgBig = vigra.sampling.resize(lab,cgp.shape)
imgBigRGB = vigra.sampling.resize(img,cgp.shape)
grad = numpy.squeeze(vigra.filters.gaussianGradientMagnitude(imgBig,2.5))+0.1


"""
print "get drag "
drag = cgp2d.DynamicRag(cgp)

print "numNodes ",drag.numberOfNodes() , cgp.numCells(2)
print "numEdges",drag.numberOfEdges()  , cgp.numCells(1)


edges =  drag.edges()

members =  inspect.getmembers(edges.__class__, predicate=inspect.ismethod)


for mem in members:
	print mem


for k,e in edges :
	print k,e



print "done"

sys.exit(0)

"""







class CgpClustering(object):
	def __init__(self,cgp):
		self.cgp = cgp 
		self.labels    = numpy.zeros(self.cgp.numCells(2),dtype=numpy.uint64)

class HierarchicalClustering(CgpClustering):
	def __init__(self,cgp):
		super(HierarchicalClustering, self).__init__(cgp)
		self.connectivity 	= cgp.sparseAdjacencyMatrix()

	def segment(self,features,nClusters):

		#print "features",features.shape
		#print "self.connectivity",self.connectivity.shape

		self.ward = WardAgglomeration(n_clusters=nClusters, connectivity=self.connectivity).fit(features.T)
		self.labels[:] = self.ward.labels_

	def mergedCgp(self):

		newLabels  = self.cgp.featureToImage(cellType=2,features=self.labels.astype(numpy.float32),useTopologicalShape=False)
		cgp,tgrid = cgp2d.cgpFromLabels(newLabels.astype(numpy.uint64)+1)
		return cgp,tgrid

print "accumulate cell feat"
feat = cgp.accumulateCellFeatures(cellType=2,image=lab,features='Mean')[0]['Mean']
feat = feat.reshape([cgp.numCells(2),-1]).astype(numpy.float32)


hc = HierarchicalClustering(cgp=cgp)
hc.segment(feat,100)
mcgp,mtgrid = hc.mergedCgp()

cgp2d.visualize(imgBigRGB,cgp=mcgp)
cgp = mcgp

print "accumulate cell feat"
feat = cgp.accumulateCellFeatures(cellType=2,image=img,features='Mean')[0]['Mean']
feat = feat.reshape([cgp.numCells(2),-1]).astype(numpy.float32)


#feat = normCProbFlat(hist)
print feat.shape


edge=cgp.cell2ToCell1Feature(feat,mode='l2')
print edge.min(),edge.max()

e1=numpy.exp(-0.00023*edge)
e0=1.0-e1
w=e1-e0




cgc,gm  = multicutFromCgp2(cgp=cgp,e0=e0,e1=e1,parameter=opengm.InfParam(planar=True,inferMinMarginals=True))
nFac    = cgp.numCells(1)
nVar    = cgp.numCells(2)


cgc.infer(cgc.verboseVisitor())

factorMinMarginals = cgc.factorMinMarginals()

print factorMinMarginals


m0 = factorMinMarginals[:,0].astype(numpy.float128)
m1 = factorMinMarginals[:,1].astype(numpy.float128)

m0*=-1.0
m1*=-1.0

p0 =  numpy.exp(m0)/(numpy.exp(m0)+numpy.exp(m1))
p1 =  numpy.exp(m1)/(numpy.exp(m0)+numpy.exp(m1))


print p1

argDual = cgc.argDual()

cgp2d.visualize(imgBigRGB,cgp=cgp,edge_data_in=p1.astype(numpy.float32))

sys.exit()



