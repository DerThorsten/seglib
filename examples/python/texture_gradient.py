import vigra
import numpy
import pylab

import seglib.edge_detectors.pixel as edp
import seglib.region_descriptors.pixel as rdp
from seglib.preprocessing import norm01

img = "/home/tbeier/src/privatOpengm/experiments/datasets/bsd500/BSR/BSDS500/data/images/test/120093.jpg"


img 	  = numpy.squeeze(vigra.readImage(img))#[0:75,0:75,:]
gradmag   = numpy.squeeze(vigra.filters.gaussianGradientMagnitude(img,sigma=2.0))
imgdt     = rdp.detexturize(img,nCluster=15,r=1,sigmaSpace=3,reductionAlg='pca',nldScale=20.0,nldEdgeThreshold=0.009,distance=None)
gradmagdt   = numpy.squeeze(vigra.filters.gaussianGradientMagnitude(imgdt,sigma=2.0))

gmult = norm01(gradmagdt)*norm01(gradmag)

gsum  = norm01(gradmag)+norm01(gradmagdt)+0.00001

f = pylab.figure()
for n, iterImg in enumerate(
	[img,gradmag,imgdt,gradmagdt,img,gmult,(2.0*gmult)/(gsum)]
	):
    #f.add_subplot(2, 1, n)  # this line outputs images on top of each other
    f.add_subplot(3, 2, n)  # this line outputs images side-by-side
    pylab.imshow(numpy.swapaxes(norm01(iterImg),0,1))
pylab.show()
