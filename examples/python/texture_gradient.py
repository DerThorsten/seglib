import vigra
import numpy
import pylab

from seglib.histogram import jointHistogram,histogram
import seglib.region_descriptors.pixel as rdp
from seglib.preprocessing import norm01,normC01


# change me to your path
img = "/home/tbeier/src/privatOpengm/experiments/datasets/bsd500/BSR/BSDS500/data/images/test/226033.jpg"


def imgStd(img,simga):

    average = vigra.filters.gaussianSmoothing(img,simga)

    return (average-img)**2


def imgStd2(img,s=5,et=2.0):

    imgd      = vigra.filters.nonlinearDiffusion(img,scale=float(s),edgeThreshold=float(et))
    return (imgd-img)**2




img       = numpy.squeeze(vigra.readImage(img))#[0:75,0:75,:]
#std       = imgStd(img,5.0)
#std2      = imgStd2(img,20.0,2.5)



def blobFinder(img):
    shape = img.shape[0:2]

    scales = numpy.arange(0.5,7.0,0.5)
    nScales = len(scales)

    bank  = numpy.zeros(shape+(nScales,))


    for si,scale in enumerate(scales):
        print scale
        log         = numpy.abs(scale*vigra.filters.laplacianOfGaussian(img,float(scale)))
        log         = numpy.sum(log,axis=2)
        bank[:,:,si]=log

        #f = pylab.figure()
        #for n, iterImg in enumerate([log,img]):
        #    f.add_subplot(1,2, n)  # this line outputs images side-by-side
        #    pylab.imshow(numpy.swapaxes(norm01(iterImg),0,1))
        #pylab.show()


    maxScale = numpy.argmax(bank,axis=2)

    print "mima",maxScale.min(),maxScale.max()
    print maxScale

    f = pylab.figure()
    for n, iterImg in enumerate([maxScale]):
        f.add_subplot(1,1, n)  # this line outputs images side-by-side
        pylab.imshow(numpy.swapaxes(norm01(iterImg),0,1))
    pylab.show()



blobFinder(img)

imgd      = vigra.filters.nonlinearDiffusion(img,scale=20.0,edgeThreshold=2.5)
gradmag   = numpy.squeeze(vigra.filters.gaussianGradientMagnitude(img,sigma=2.0))
gradmagd  = numpy.squeeze(vigra.filters.gaussianGradientMagnitude(imgd,sigma=2.0))

f = pylab.figure()
for n, iterImg in enumerate(
    [img,gradmag,imgd,gradmagd]
    ):
    #f.add_subplot(2, 1, n)  # this line outputs images on top of each other
    f.add_subplot(2, 2, n)  # this line outputs images side-by-side
    pylab.imshow(numpy.swapaxes(norm01(iterImg),0,1))
pylab.show()



hist  = histogram(imgd,bins=30,r=1,sigma=[float(3.0),float(1.0)])
#hist  = jointHistogram(img,bins=5,r=1,sigma=[float(1.0),float(1.0)])

imgdt     = rdp.detexturize(hist,nCluster=15,reductionAlg='pca',nldScale=10.0,nldEdgeThreshold=0.000005,distance=None)
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
