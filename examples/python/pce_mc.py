import vigra
import numpy
import opengm
from seglib import cgp2d 
from seglib.clustering.ce_multicut import *


img = "img/37073.jpg"
img = "img/42049.jpg"
img = numpy.squeeze(vigra.readImage(img))#[0:75,0:75,:]
lab = vigra.colors.transform_RGB2Lab(img)
labels ,nseg= vigra.analysis.slicSuperpixels(lab,10.0,25)
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


#cgc.infer(cgc.verboseVisitor())



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

        j3 = []
        j4 = []
        # find all 3 and  4 junctions
        for cell0 in cgp.cells(0):
            bounds = numpy.array(cell0.bounds)-1
            nb = len(bounds)
            if nb == 3:
                j3.append(bounds)
            if nb == 4:
                j4.append(bounds)
        self.j3=numpy.array(j3)
        self.j4=numpy.array(j4)

        self.j3S=numpy.zeros(self.j3.shape,dtype=numpy.uint64)
        self.j4S=numpy.zeros(self.j4.shape,dtype=numpy.uint64)

        print "shapes ",self.j3.shape,self.j4.shape
        print "shapes ",self.j3S.shape,self.j4S.shape

    def __call__(self,argPrimal,argDual):
        where1    = numpy.where(argDual==1)
        unaryCost = numpy.sum(self.weights[where1])


        binCount = numpy.bincount(argPrimal.astype(numpy.int32))

        cost = 1.0/binCount

        sizeCost = numpy.sum(cost)



        for c in range(3):
            self.j3S[:,c] =  argDual[self.j3[:,c]]
        for c in range(4):
            self.j4S[:,c] = argDual[self.j4[:,c]]


        jsum = numpy.sum(self.j3S,axis=1)
        sum3 = numpy.sum(jsum==3)
        #print "sum3",sum3

        jsum = numpy.sum(self.j3S,axis=1)
        sum4 = numpy.sum(jsum<=3)
        #print "sum4",sum4

        #print "j4S",self.j4S



        #print "size cost ",sizeCost
        #print "uanry cost",unaryCost


        return unaryCost+0.5*sizeCost+0.5*sum3 + 0.5*sum4



obj = Objective(cgp,initWeights)


nSamples = 15
nElites  = 1


mixLow    = 0.1
mixHigh   = 0.7

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
    #print "offset ",mOffset,varOffset
    foundBest=False
    for sampleIndex in range(nSamples):
        # sample new weights
        offset  = gaussOffset(0.0,0.025)
        weights = sampleFromGauss(mean=weightsMean,std=weightsStd,out=weights) + offset
        # update multicut weights
        cgc.changeWeights(weights)

        # infer
        print  sampleIndex,iteration
        #cgc.infer(cgc.verboseVisitor())
        cgc.infer()
        argDual[sampleIndex,:]   = cgc.argDual(out=argDual[sampleIndex,:]) 
        argPrimal[sampleIndex,:] = cgc.arg() 
        objVal[sampleIndex] = obj(argPrimal=argPrimal[sampleIndex,:],argDual=argDual[sampleIndex,:])
 
        weightSamples[sampleIndex,:] = weights[:]
        weightOffset[sampleIndex] = offset
        #print "objval ",objVal[sampleIndex],"bestObj",bestObj

        if objVal[sampleIndex] < bestObj:
            foundBest=True
            bestObj=objVal[sampleIndex]
            print "bestObj",bestObj 
            bestState=argDual[sampleIndex,:].copy()

            
        if iteration%100==0 and sampleIndex==0:
            imgBigRgb = vigra.sampling.resize(img,cgp.shape)
            cgp2d.visualize(imgBigRgb,cgp,edge_data_in=bestState.astype(numpy.float32))


    if foundBest :
        mix = mixHigh
    else:
        mix = mixLow
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

    #amax = numpy.abs(weightsMean).max()
    #weightsMean/=amax

    #amax = numpy.abs(weightsMeanNew).max()
    #weightsMeanNew/=amax


    weightsMean =  mix * weightsMeanNew    + (1.0-mix)*weightsMean
    weightsStd  =  mix * weightsStdNew     + (1.0-mix)*weightsStd


   



    #print  "wminmax ",weightsMean.min(),weightsMean.max()