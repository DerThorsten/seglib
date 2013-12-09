import vigra
import numpy
import opengm
from seglib import cgp2d 
from seglib.clustering.ce_multicut import *


img = "img/37073.jpg"
img = "img/42049.jpg"
img = numpy.squeeze(vigra.readImage(img))#[0:75,0:75,:]
lab = vigra.colors.transform_RGB2Lab(img)
labels ,nseg= vigra.analysis.slicSuperpixels(lab,10.0,15)
labels = vigra.analysis.labelImage(labels).astype(numpy.uint64)
cgp,tgrid = cgp2d.cgpFromLabels(labels)
imgBig = vigra.sampling.resize(lab,cgp.shape)
grad = numpy.squeeze(vigra.filters.gaussianGradientMagnitude(imgBig,4.5))+0.1


if False :
    imgBigRgb = vigra.sampling.resize(img,cgp.shape)
    cgp2d.visualize(imgBigRgb,cgp)


weights = imgToWeight(cgp=cgp,img=grad,gamma=3.5,method='exp')
cgc,gm  = multicutFromCgp(cgp=cgp,weights=weights)
nFac    = cgp.numCells(1)
nVar    = cgp.numCells(2)


cgc.infer()



argDual     = numpy.ones(nFac,dtype=opengm.label_type)
argDual=cgc.argDual(out=argDual)
argDual=argDual.astype(numpy.float32)
ce = CeMc(cgp=cgp)


if False :
    imgBigRgb = vigra.sampling.resize(img,cgp.shape)
    cgp2d.visualize(imgBigRgb,cgp,edge_data_in=argDual)






initWeights = weights.copy()
weightsMean = weights.copy()
weightsStd  = numpy.ones(nFac,dtype=numpy.float32)*0.25







class Objective(object):
    def __init__(self,cgp,weights):
        self.cgp     = cgp
        self.weights = weights

    def __call__(self,argPrimal,argDual):
        where1    = numpy.where(argDual==1)
        unaryCost = numpy.sum(self.weights[where1])


        binCount = numpy.bincount(argPrimal.astype(numpy.int32))

        cost = 1.0/binCount

        sizeCost = numpy.sum(cost)

        #print "size cost",sizeCost

        #print unaryCost
        return unaryCost+0.5*sizeCost



obj = Objective(cgp,initWeights)


nSamples = 15
nElites  = 2
mix      = 0.05

mOffset   = 0.0
varOffset = 0.1


bestState     = numpy.zeros([nVar],dtype=opengm.label_type)

bestObj = float('inf')
for  iteration in range(1000):

    argDual       = numpy.zeros([nSamples,nFac],dtype=opengm.label_type)
    argPrimal     = numpy.zeros([nSamples,nVar],dtype=opengm.label_type)
    objVal        = numpy.zeros([nSamples],dtype=opengm.value_type)
    weightSamples = numpy.zeros([nSamples,nFac],dtype=opengm.value_type)
    weightOffset  = numpy.zeros([nSamples],dtype=opengm.value_type)
    
    # FILL SAMPLES
    print "offset ",mOffset,varOffset
    for sampleIndex in range(nSamples):
        # sample new weights
        offset  = gaussOffset(0.0,0.1)
        weights = sampleFromGauss(mean=weightsMean,std=weightsStd,out=weights) + offset
        # update multicut weights
        cgc.changeWeights(weights)

        # infer
        print  sampleIndex
        #cgc.infer(cgc.verboseVisitor())
        cgc.infer()
        argDual[sampleIndex,:]   = cgc.argDual(out=argDual[sampleIndex,:]) 
        argPrimal[sampleIndex,:] = cgc.arg() 
        objVal[sampleIndex] = obj(argPrimal=argPrimal[sampleIndex,:],argDual=argDual[sampleIndex,:])
 
        weightSamples[sampleIndex,:] = weights[:]
        weightOffset[sampleIndex] = offset
        print "objval ",objVal[sampleIndex],"bestObj",bestObj

        if objVal[sampleIndex] < bestObj:
            bestObj=objVal[sampleIndex]
            bestState=argDual[sampleIndex,:].copy()

            print "zeros",numpy.sum( bestState==0)
            print "ones ",numpy.sum( bestState==1)


        if iteration%40==0 and sampleIndex==0:
            imgBigRgb = vigra.sampling.resize(img,cgp.shape)
            print "zeros",numpy.sum( bestState==0)
            print "ones ",numpy.sum( bestState==1)
            cgp2d.visualize(imgBigRgb,cgp,edge_data_in=bestState.astype(numpy.float32))


    # get best samples
    sortedIndex = numpy.argsort(objVal)
    eliteArgDual = argDual[sortedIndex[:nElites]]

    # new mean
    weightsMeanNew= numpy.mean( weightSamples[sortedIndex[:nElites]],axis=0)
    weightsStdNew = numpy.std( weightSamples[sortedIndex[:nElites]],axis=0)

    whereSmall = numpy.where(weightsStdNew<0.2)
    weightsStdNew[whereSmall]=0.2

    # new offset 
    newOffsetMean  = numpy.mean(weightOffset[sortedIndex[:nElites]])
    newOffsetStd  = numpy.std(weightOffset[sortedIndex[:nElites]])


    mOffset =  mix * newOffsetMean    + (1.0-mix)*mOffset
    varOffset  =  mix * newOffsetStd     + (1.0-mix)*varOffset


    weightsMean =  mix * weightsMeanNew    + (1.0-mix)*weightsMean
    weightsStd  =  mix * weightsStdNew     + (1.0-mix)*weightsStd