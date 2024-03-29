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
labels ,nseg= vigra.analysis.slicSuperpixels(lab,10.0,10)
labels = vigra.analysis.labelImage(labels).astype(numpy.uint64)
cgp,tgrid = cgp2d.cgpFromLabels(labels)
imgBig = vigra.sampling.resize(lab,cgp.shape)
imgBigRGB = vigra.sampling.resize(img,cgp.shape)
grad = numpy.squeeze(vigra.filters.gaussianGradientMagnitude(imgBig,2.5))+0.1

print cgp.numCells(2),cgp.numCells(1)
sys.exit()

print "accumulate cell feat"
feat = cgp.accumulateCellFeatures(cellType=2,image=lab,features='Mean')[0]['Mean']
feat = feat.reshape([cgp.numCells(2),-1]).astype(numpy.float32)

hc = HierarchicalClustering(cgp=cgp)
hc.segment(feat,100)
mcgp,mtgrid = hc.mergedCgp()

cgp2d.visualize(imgBigRGB,cgp=mcgp)
cgp = mcgp

print "accumulate cell feat"
feat = cgp.accumulateCellFeatures(cellType=2,image=lab,features='Mean')[0]['Mean']
feat = feat.reshape([cgp.numCells(2),-1]).astype(numpy.float32)


#feat = normCProbFlat(hist)
print feat.shape


edge=cgp.cell2ToCell1Feature(feat,mode='l2')
print edge.min(),edge.max()

e1=numpy.exp(-0.00023*edge)
e0=1.0-e1
w=e1-e0




cgc,gm  = multicutFromCgp2(cgp=cgp,e0=e0,e1=e1,parameter=opengm.InfParam(planar=True,inferMinMarginals=False))
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

sys.exit()



