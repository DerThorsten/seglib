from seglib.histogram import jointHistogram,histogram
from seglib.region_descriptors.pixel import imageCodebookClustering,imageDimensionReduction
from seglib.preprocessing import norm01,normC01

import pylab
import numpy

import vigra


def textureGradient(
	img,
	nCluster=15
):
    shape 	= img.shape
    print "get histogram"
    # get histogram
    hist = histogram(img,r=1,sigma=[4.0,2.0])

    print "codebook clustering"
    # fuzzy cluster in codebooks
    labels,centers,distToCenters = imageCodebookClustering(
    img=hist,nClusters=15,distance='bhattacharyya',batchSize=1000,iterations=500,nInit=3
    )

    print "do smothing"
    smoothed = numpy.ones(shape[0:2] + (nCluster,))

    for k in range(15):

        cImg = distToCenters[:,:,k].copy().astype(numpy.float32)
        #pylab.imshow(numpy.swapaxes(distToCenters[:,:,k],0,1))
        #pylab.show()
        sImg = vigra.filters.nonlinearDiffusion(cImg,scale=20.0,edgeThreshold=0.01)
        smoothed[:,:,k]=sImg

    print "dimension reduction"
    
    

    #pylab.imshow(numpy.swapaxes(sImg,0,1))
    #pylab.show()


    reducted = imageDimensionReduction(smoothed,nComponents=3,alg='pca')

    print "reductedshape",reducted.shape

    pylab.imshow(numpy.swapaxes(normC01(reducted),0,1))
    pylab.show()

