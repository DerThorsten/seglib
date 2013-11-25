import vigra
import numpy
import pylab

import seglib.edge_detectors.pixel as edp
import seglib.region_descriptors.pixel as rdp
from seglib.preprocessing import norm01
from seglib.histogram import jointHistogram,histogram
from seglib.region_descriptors.pixel.sift import denseSift


img = "img/text.jpg"
img = "img/img.png"

img 	  = numpy.squeeze(vigra.readImage(img))#[0:75,0:75,:]
gradmag   = numpy.squeeze(vigra.filters.gaussianGradientMagnitude(img,sigma=4.0))


sift  = denseSift(img)
imgdt = rdp.deepDetexturize(srcImg=img,img=sift,nIteration=10,
	nCluster=10,reductionAlg='pca',nldEdgeThreshold=0.005,nldScale=10.0,distance=None)#'cityblock')







f = pylab.figure()
for n, iterImg in enumerate(
	[img,imgdt]
	):
    #f.add_subplot(2, 1, n)  # this line outputs images on top of each other
    f.add_subplot(1, 2, n)  # this line outputs images side-by-side
    pylab.imshow(numpy.swapaxes(norm01(iterImg),0,1))
pylab.show()

hist  = histogram(imgdt,bins=30,r=1,sigma=[float(3.0),float(1.0)])

imgdt     = rdp.detexturize(hist,nCluster=15,reductionAlg='pca',nldScale=20.0,nldEdgeThreshold=0.3,distance=None)
gradmagdt   = numpy.squeeze(vigra.filters.gaussianGradientMagnitude(imgdt,sigma=2.0))

gmult = norm01(gradmagdt)*norm01(gradmag)

gsum  = norm01(gradmag)+norm01(gradmagdt)+0.00001

f = pylab.figure()
for n, iterImg in enumerate(
	[img,gradmagdt,imgdt,gradmag]
	):
    #f.add_subplot(2, 1, n)  # this line outputs images on top of each other
    f.add_subplot(2, 2, n)  # this line outputs images side-by-side
    if iterImg.ndim==2:
    	pylab.imshow(numpy.swapaxes(norm01(iterImg),0,1),cmap='gray')
    else :
    	pylab.imshow(numpy.swapaxes(norm01(iterImg),0,1))
pylab.show()
