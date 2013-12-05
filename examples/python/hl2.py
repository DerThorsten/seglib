import vigra
import numpy
import opengm
from seglib import cgp2d 
from seglib.clustering.ce_multicut import *


img = "img/37073.jpg"
img = "img/42049.jpg"
binCount=15
sigma = 1.5
img = numpy.squeeze(vigra.readImage(img))#[0:75,0:75,:]
lab = vigra.colors.transform_RGB2Lab(img)
labels ,nseg= vigra.analysis.slicSuperpixels(lab,10.0,25)
labels = vigra.analysis.labelImage(labels).astype(numpy.uint64)
cgp,tgrid = cgp2d.cgpFromLabels(labels)
imgBig = vigra.sampling.resize(lab,cgp.shape)
grad = numpy.squeeze(vigra.filters.gaussianGradientMagnitude(imgBig,4.5))+0.1


print "accumulate cell hist"
hist = cgp.accumulateCellHistogram(cellType=2,image=img,binCount=binCount,sigma=sigma)
hist = hist.reshape([cgp.numCells(2),-1]).astype(numpy.float32)
print hist.shape
#hist=vigra.taggedView(hist,"xc")
#hist=hist.transposeToVigraOrder()

hist=numpy.array(hist)
print "construkt"
hlo = cgp2d.HighLevelObjective(cgp)

print "set features"
hlo.setRegionFeatures(hist)


