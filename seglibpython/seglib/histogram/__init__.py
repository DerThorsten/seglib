from _histogram import *
import numpy
import vigra
from scipy.ndimage.filters import gaussian_filter1d as GaussianFilter1d



"""
def gaussianSmoothing1d(histograms,sigma):
  nHist = histograms.shape[0]
  nBins = histograms.shape[1]

  kernel=vigra.filters.Kernel1D()
  kernel.initDiscreteGaussian(sigma)

  smoothedHistograms=vigra.filters.convolveOneDimension(histograms, dim=1, kernel=kernel)

  return smoothedHistograms
"""
from scipy.ndimage.filters import  gaussian_filter as GaussainFilter


def histogram(image,dmin=None,dmax=None,bins=32,r=3,sigma=[2.0,1.0],out=None):


    img       = numpy.require(image,dtype=numpy.float32)
    nChannels = img.shape[2]
    flat      = img.reshape([-1,nChannels])

    if dmin is None :
        dmin = numpy.require(numpy.min(flat,axis=0),dtype=numpy.float32)
    if dmax is None :
        dmax = numpy.require(numpy.max(flat,axis=0),dtype=numpy.float32)

    
    # computet the actual histogram
    rawHist  = _histogram._batchHistogram_( img=img,dmin=dmin,dmax=dmax,bins=bins,r=r,out=out)



    if sigma is None :
        return rawHist

    else : 
        if isinstance(sigma,(float,int,long)):
            sigmas = [sigma]
        else :
            sigmas = sigma
        assert len(sigmas)<=2 


        if len(sigmas)==1 :
            # only smooth bins
            for c in range(nChannels):
                cHist = rawHist[:,:,c,:]
                kernel=vigra.filters.Kernel1D()
                kernel.initDiscreteGaussian(float(sigmas[0]))
                #kernel.setBorderTreatment()
                smoothedHistograms=vigra.filters.convolveOneDimension(cHist, dim=2, kernel=kernel)
                rawHist[:,:,c,:] = smoothedHistograms[:,:,:]
            return rawHist
        if len(sigmas)==2 :
            # smooth bins ans spatial
            for c in range(nChannels):
                cHist = rawHist[:,:,c,:]
                
                s = [sigmas[1]]*2 + [sigmas[0]]

                smoothedHistograms = GaussainFilter(cHist,sigma=s,order=0)#,mode='constant',cval=0.0)
                rawHist[:,:,c,:] = smoothedHistograms[:,:,:]
            return rawHist



def batchJointHistogram(img,r=1,bins=5,sigma=[1.0,1.0]):
    nCp      = img.shape[2]/3
    outShape = [img.shape[0],img.shape[1],nCp,bins,bins,bins]

    out      = numpy.zeros(outShape,dtype=numpy.float32)


    for cp in range(nCp): 
        inputImg = img[:,:,cp*3:(cp+1)*3]
        cOut = out[:,:,cp,:,:,:]
        cOut = jointHistogram(image=inputImg,bins=bins,r=r,sigma=sigma,out=cOut)


    return out




def jointHistogram(image,dmin=None,dmax=None,bins=5.0,r=1,sigma=[1.0,1.0],out=None):
    #img       = numpy.require(image,dtype=numpy.float32)
    img = image
    nChannels = img.shape[2]
    flat      = img.reshape([-1,nChannels])

    #print "flatshape",flat.shape

    assert nChannels == 3

    if dmin is None :
        dmin = numpy.require(numpy.min(flat,axis=0),dtype=numpy.float32)
        dmin = [float(dmin[x]) for x in range(3)]
    if dmax is None :
        dmax = numpy.require(numpy.max(flat,axis=0),dtype=numpy.float32)
        dmax = [float(dmax[x]) for x in range(3)]
    b = (bins,bins,bins)

    #print dmin
    #print dmax

    imgHist = _histogram._jointColorHistogram_(img=img,dmin=dmin,dmax=dmax,bins=b,r=r,out=out)


    if sigma is not None :
        s = sigma[1]*2 + sigma[0]*3
        imgHist = GaussainFilter(imgHist,sigma=s,order=0)
    return imgHist



def labelHistogram(img,nLabels,r=1,sigma=1.0,out=None,visu=False):

    nInputLabelings = img.shape[2]
    labels = numpy.require(img,dtype=numpy.uint64)


    labelHist = _histogram._label_histogram_(img=labels,nLabels=long(nLabels),r=long(r),out=out)





    
    # convolce along x and y axis  ( 0 and 1 )
    labelHistTmp  = labelHist.copy()
    labelHistTmp = GaussianFilter1d(labelHist,     sigma=sigma, axis=0, order=0, output=None, mode='reflect', cval=0.0)
    labelHist = GaussianFilter1d(labelHistTmp,  sigma=sigma, axis=1  , order=0, output=None, mode='reflect', cval=0.0)


    #print "difference",numpy.sum(numpy.abs(labelHistTmp-labelHist))
    
    if visu :
        import pylab
        import matplotlib
        cmap = matplotlib.colors.ListedColormap ( numpy.random.rand ( 256,3))
        for i in range(nInputLabelings):
            for l in range(nLabels):

                limg = labelHist[:,:,i,l]
                pylab.imshow ( numpy.swapaxes(limg,0,1), cmap = "jet")
                pylab.show()

    return labelHist




def labelHistogramNew(img,nLabels,labelSim,r=1,sigma=1.0,out=None,visu=False):

    dx,dy = img.shape[0:2]

    ndim = img.ndim 

    # a single labeling
    if ndim == 2 :
        assert labelSim.ndim==2
        assert labelSim.shape[0]==nLabels
        assert labelSim.shape[1]==nLabels
        img  = img.reshape([dx,dy,1])
        labelSim=labelSim.reshape([nLabels,nLabels,1])
    # multiple labelings
    elif ndim == 3:
        assert labelSim.ndim==3
        assert labelSim.shape[0]==img.shape[2]
        assert labelSim.shape[1]==nLabels
        assert labelSim.shape[2]==nLabels
    else :
        raise RuntimeError("""img.ndim must be either 2 (a single labeling) or 3 (multiple labelings).
            the axis ordering must be (x,y) or (x,y,bins)
            """); 

    img      = numpy.require(img,dtype=numpy.uint64)
    labelSim = numpy.require(labelSim,dtype=numpy.float32)

    print img.shape,labelSim.shape

    print labelSim[1,2,0],labelSim[2,1,0]

    """
    labelSim[:]=0.0
    for l  in range(nLabels ):
        labelSim[l,l,0]=1.0
    """
    labelHist=_histogram._label_sim_histogram_(
        img=img.copy(),
        labelSim=labelSim.copy(),
        nLabels=nLabels,
        r=r
    )
    if sigma is not None  and sigma >=0.05:
        # convolce along x and y axis  ( 0 and 1 )
        labelHistTmp = labelHist.copy()
        labelHistTmp = GaussianFilter1d(labelHist,     sigma=sigma, axis=0, order=0, output=None, mode='reflect', cval=0.0)
        labelHist    = GaussianFilter1d(labelHistTmp,  sigma=sigma, axis=1  , order=0, output=None, mode='reflect', cval=0.0)


    if visu :
        import pylab
        import matplotlib
        cmap = matplotlib.colors.ListedColormap ( numpy.random.rand ( 256,3))
        for i in range(img.shape[2]):
            for l in range(nLabels):

                limg = labelHist[:,:,i,l]
                print "mima",limg.min(),limg.max()
                pylab.imshow ( numpy.swapaxes(limg,0,1), cmap = "jet")
                pylab.show()

    return labelHist



def centerDistToBinSimilarity(centers,norm,gamma,truncate=0.000001):

    c = centers.copy()
    c-=c.min()
    c/=c.max()
    k = centers.shape[0]
    f = centers.shape[1]

    #Print "k",k,"f",f

    diffarray  = numpy.zeros([k,k],dtype=numpy.float32)



    for k1 in range(k-1):
        for k2 in range(k1+1,k):
            d = numpy.sum(numpy.abs(centers[k1,:]-centers[k2,:])**norm)
            #print k1,k2,"diffssss",d
            diffarray[k1,k2]=d
            diffarray[k2,k1]=d


    r = numpy.exp(-gamma*diffarray)



    for kk in range(k):
        r[kk,kk]=1.0

    for kk in range(k):
        r[kk,:]=r[kk,:]/numpy.sum(r[kk,:])
    #rint r

    r[r<truncate]=0.0

    for kk in range(k):
        r[kk,:]=r[kk,:]/numpy.sum(r[kk,:])
    #print r


    #for k1 in range(k-1):
    #    print k1,k1,"diffssss",r[k1,k1],"d",diffarray[k1,k1]
    #    for k2 in range(k1+1,k):
    #        print k1,k2,"diffssss",r[k1,k2],"d",diffarray[k1,k2]

    return r