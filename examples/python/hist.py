import vigra
import numpy
import pylab

from seglib.histogram import jointHistogram,histogram
import seglib.region_descriptors.pixel as rdp
from seglib.preprocessing import norm01,normC01,normCProb,reshapeToImage,reshapeFromImage


# change me to your path
img = "/home/tbeier/src/privatOpengm/experiments/datasets/bsd500/BSR/BSDS500/data/images/test/226033.jpg"
img       = numpy.squeeze(vigra.readImage(img))#[0:75,0:75,:]
hist  = histogram(img,bins=50,r=1,sigma=[float(0.5),float(0.5)])
hist =reshapeToImage(hist,img.shape)

hist =normCProb(hist)

nBins = hist.shape[2]
print "bins",nBins


flatHist = reshapeFromImage(hist).copy()
l = numpy.log2(flatHist)
l = numpy.nan_to_num(l)


e = flatHist*l
e = numpy.sum(e,axis=1)*(-1.0)
print "eshape",e.shape
eimg =numpy.squeeze(reshapeToImage(e,img.shape).copy())
print "eimgshape",eimg.shape





f = pylab.figure()
for n, iterImg in enumerate([img,eimg]):
    #f.add_subplot(2, 1, n)  # this line outputs images on top of each other
    f.add_subplot(1, 2, n)  # this line outputs images side-by-side
    pylab.imshow(numpy.swapaxes(norm01(iterImg),0,1))
pylab.show()
