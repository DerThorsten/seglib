import pylab
import numpy
import vigra

from seglib.histogram import jointHistogram,histogram,centerDistToBinSimilarity,labelHistogramNew
from seglib.region_descriptors.pixel import imageCodebookClustering,imageDimensionReduction
from seglib.preprocessing import norm01,normC01


def detexturize(
	img,
	nCluster=15,
    reductionAlg='pca',
    distance=None,
    nldScale=20.0,
    nldEdgeThreshold=0.01
):
    shape 	= img.shape
    print "codebook clustering"
    # fuzzy cluster in codebooks
    labels,centers,distToCenters = imageCodebookClustering(
    img=img,nClusters=nCluster,distance=distance,batchSize=1000,iterations=500,nInit=3
    )

    if False:
        print "labelss",labels.shape

        pylab.imshow(numpy.swapaxes(labels,0,1))
        pylab.show()


        print "center simmilarity"
        sim = centerDistToBinSimilarity(centers=centers,norm=1,gamma=4.0)

        labelHist  = labelHistogramNew(labels,nLabels=nCluster,labelSim=sim,r=1,sigma=1.0,visu=True)

        print sim

        print "compute a label histogram"


    print "do smothing"
    smoothed = numpy.ones(shape[0:2] + (nCluster,))

    for k in range(nCluster):
        print k
        cImg = distToCenters[:,:,k].copy().astype(numpy.float32)
        #pylab.imshow(numpy.swapaxes(distToCenters[:,:,k],0,1))
        #pylab.show()
        sImg = vigra.filters.nonlinearDiffusion(cImg,scale=nldScale,edgeThreshold=nldEdgeThreshold)
        smoothed[:,:,k]=sImg

    print "dimension reduction"
    
    

    #pylab.imshow(numpy.swapaxes(sImg,0,1))
    #pylab.show()


    reducted = imageDimensionReduction(smoothed,nComponents=3,alg=reductionAlg).astype(numpy.float32)
    reducted = vigra.taggedView(reducted, 'xyc')

    

    #Apylab.imshow(numpy.swapaxes(normC01(reducted),0,1))
    #pylab.show()

    return reducted




def deepDetexturize(
    srcImg,
    img,
    nIteration=10,
    **kwargs
):
    hist=img.copy()
    mixIn=None
    for i in range(nIteration):

        newImg = detexturize(img=hist,**kwargs)
        newImgIter = newImg.copy()
        newImgIter = vigra.taggedView(newImgIter,'xyc')
        if i == 0:
            mixIn=newImg.copy()
        if i !=0 :
            newImg = numpy.concatenate([newImg,mixIn],axis=2)
            newImg     = vigra.taggedView(newImg,'xyc')
        hist   = histogram(newImg,r=2,sigma=[3.0,3.0])

        f = pylab.figure()
        for n, iterImg in enumerate([srcImg,newImgIter]):
            #f.add_subplot(2, 1, n)  # this line outputs images on top of each other
            f.add_subplot(1, 2, n)  # this line outputs images side-by-side
            if iterImg.ndim==2:
                pylab.imshow(numpy.swapaxes(norm01(iterImg),0,1),cmap='gray')
            else :
                pylab.imshow(numpy.swapaxes(norm01(iterImg),0,1))
        pylab.show()