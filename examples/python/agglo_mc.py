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
from seglib.region_descriptors.pixel.sift import denseSift
from seglib.histogram import jointHistogram,histogram

img = "img/37073.jpg"
img="img/42049.jpg"
img="/home/tbeier/src/seglib/examples/python/img/lena.bmp"
img="/home/tbeier/src/seglib/examples/python/img/t.jpg"
img="/home/tbeier/src/privatOpengm/experiments/100075.jpg"
#img="img/zebra.jpg"
img = numpy.squeeze(vigra.readImage(img))#[0:75,0:75,:]
lab = vigra.colors.transform_RGB2Lab(img)
labels ,nseg= vigra.analysis.slicSuperpixels(lab,10.0,5)
labels = vigra.analysis.labelImage(labels).astype(numpy.uint64)
cgp,tgrid = cgp2d.cgpFromLabels(labels)
imgBig = vigra.sampling.resize(lab,cgp.shape)
imgBigRGB = vigra.sampling.resize(img,cgp.shape)
grad = numpy.squeeze(vigra.filters.gaussianGradientMagnitude(imgBig,2.0))


hist = histogram(lab,bins=30,r=2,sigma=[2.0,3.0])
hist = normCProb(reshapeToImage(hist,lab.shape))

#pylab.imshow(hist[:,:,0])
#pylab.show()
print cgp.numCells(2),cgp.numCells(1)

siftImg = denseSift(lab[:,:,0])
siftImg = normCProb(siftImg)

if True:
	print "accumulate cell feat"
	feat = cgp.accumulateCellFeatures(cellType=2,image=lab,features='Mean')[0]['Mean']
	feat = feat.reshape([cgp.numCells(2),-1]).astype(numpy.float32)

	hc = HierarchicalClustering(cgp=cgp)
	hc.segment(feat,300)
	mcgp,mtgrid = hc.mergedCgp()


	cgp = mcgp

print cgp.numCells(2),cgp.numCells(1)



aggloCut = AggloCut(initCgp=cgp,edgeImage=grad,featureImage=imgBig,rgbImage=imgBigRGB,siftImage=siftImg,histImage=hist)
aggloCut.infer(gammas=[0.2],deleteN=1)






sys.exit(1)





cgc,gm 	= multicutFromCgp(cgp=cgp,weights=weights,parameter=opengm.InfParam(planar=True,inferMinMarginals=True))
#cgc,gm  = multicutFromCgp2(cgp=cgp,e0=e0,e1=e1,parameter=opengm.InfParam(planar=True,inferMinMarginals=False))
nFac    = cgp.numCells(1)
nVar    = cgp.numCells(2)


cgc.infer(cgc.verboseVisitor())
argDual = cgc.argDual()
cgp2d.visualize(imgBigRGB,cgp=cgp,edge_data_in=argDual.astype(numpy.float32))




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

sys.exit(1)



