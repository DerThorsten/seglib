import vigra
import numpy
import pylab

from seglib import cgp2d 
from seglib.preprocessing import norm01
import seglib.edge_detectors.pixel as edp
import seglib.region_descriptors.pixel as rdp
from seglib.preprocessing import norm01
from seglib.histogram import jointHistogram,histogram
from seglib.region_descriptors.pixel.sift import denseSift


# change me to your path
img = "/home/tbeier/src/privatOpengm/experiments/datasets/bsd500/BSR/BSDS500/data/images/test/226033.jpg"
img = numpy.squeeze(vigra.readImage(img))#[0:75,0:75,:]

binCount = 30 
sigma = 1.5
histImg = numpy.zeros(img.shape[0:2]+(binCount*3,))

imgBig = None

sizes = [3,4,5,8,10,15,20,25,40,100]
scalings = [5,10,15]


for size in  sizes:
    for scaling in scalings:

        size = int (size)
        scaling = float(scaling)

        print size,scaling

        labels ,nseg= vigra.analysis.slicSuperpixels(vigra.colors.transform_RGB2Lab(img),scaling,size)
        labels = vigra.analysis.labelImage(labels).astype(numpy.uint64)
        cgp,tgrid = cgp2d.cgpFromLabels(labels)

        if imgBig is None:
            imgBig=vigra.sampling.resize(img,cgp.shape)
        #cgp2d.visualize(imgBig,cgp=cgp)


        print "accumulate cell "
        hist = cgp.accumulateCellHistogram(cellType=2,image=img,binCount=binCount,sigma=sigma)
        hist = hist.reshape([cgp.numCells(2),-1])



        for c in range(histImg.shape[2]):
            histImg[:,:,c] += (size)*cgp.featureToImage(cellType=2,features=hist[:,c],ignoreInactive=False,useTopologicalShape=False)

histImg=numpy.require(histImg,dtype=numpy.float32)
histImg=vigra.taggedView(histImg, 'xyc')


histImg = vigra.gaussianSmoothing(histImg,sigma=1.0)

#for c in range(histImg.shape[2]):
#    #print c
#    pylab.imshow( numpy.swapaxes( norm01(histImg[:,:,c]) ,0,1) )
#    pylab.show()
#
#    print "hist",hist.shape


imgdt = rdp.deepDetexturize(srcImg=img,img=histImg,nIteration=10,
    nCluster=10,reductionAlg='pca',nldEdgeThreshold=10.0,nldScale=10.0,distance=None)#'cityblock')
