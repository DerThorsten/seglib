import vigra
import numpy
import pylab

import seglib.edge_detectors.pixel as edp
import seglib.region_descriptors.pixel as rdp
from seglib.preprocessing import norm01,normCProb,reshapeToImage
from seglib.histogram import jointHistogram,histogram
from seglib.region_descriptors.pixel.sift import denseSift


img = "img/text.jpg"
img = "img/108005.jpg"
img="img/zebra.jpg"
img 	  = numpy.squeeze(vigra.readImage(img))#[0:75,0:75,:]
gradmag   = numpy.squeeze(vigra.filters.gaussianGradientMagnitude(img,sigma=2.0))


images  = [img, vigra.colors.transform_RGB2Lab(img), vigra.colors.transform_RGB2Luv(img)]

hists = [  normCProb(reshapeToImage(histogram( i,r=3,sigma=[3.0,3.0]),img.shape))  for i in images]

hists.append(normCProb(denseSift(img)))

imgdt = rdp.detexturize2(hists,nCluster=15,reductionAlg='pca',nldEdgeThreshold=0.001,nldScale=15.0,distance=None)#'cityblock')






f = pylab.figure()
for n, iterImg in enumerate(
	[img,imgdt]
	):
    #f.add_subplot(2, 1, n)  # this line outputs images on top of each other
    f.add_subplot(1, 2, n)  # this line outputs images side-by-side
    pylab.imshow(numpy.swapaxes(norm01(iterImg),0,1))
pylab.show()

hist  = histogram(imgdt,bins=15,r=1,sigma=[float(3.0),float(1.0)])

imgdt     = rdp.detexturize(hist,nCluster=15,reductionAlg='pca',nldScale=20.0,nldEdgeThreshold=0.015,distance=None)
gradmagdt   = numpy.squeeze(vigra.filters.gaussianGradientMagnitude(imgdt,sigma=2.0))

gmult = norm01(gradmagdt)*norm01(gradmag)

gsum  = norm01(gradmag)+norm01(gradmagdt)+0.00001

f = pylab.figure()
for n, iterImg in enumerate(
	[img,gradmagdt,imgdt,gradmag,gmult,gmult]
	):
    #f.add_subplot(2, 1, n)  # this line outputs images on top of each other
    f.add_subplot(2, 3, n)  # this line outputs images side-by-side
    if iterImg.ndim==2:
    	pylab.imshow(numpy.swapaxes(norm01(iterImg),0,1),cmap='gray')
    else :
    	pylab.imshow(numpy.swapaxes(norm01(iterImg),0,1))
pylab.show()
