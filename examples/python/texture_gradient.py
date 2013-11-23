import vigra
import numpy
import pylab

import seglib.edge_detectors.pixel as edp
import seglib.region_descriptors.pixel as rdp
from seglib.preprocessing import norm01

img 	  = numpy.squeeze(vigra.readImage('zebra.jpg'))#[0:75,0:75,:]
gradmag   = numpy.squeeze(vigra.filters.gaussianGradientMagnitude(img,sigma=4.0))
imgdt     = rdp.detexturize2(img,nCluster=8,r=1,sigmaSpace=2,reductionAlg='pca',nldScale=3.0,distance=None)#'cityblock')
imgdt     = rdp.detexturize(imgdt,nCluster=15,r=1,sigmaSpace=5,reductionAlg='pca',nldScale=3.0,distance=None)
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
