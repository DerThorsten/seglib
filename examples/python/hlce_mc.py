import vigra
import numpy
import opengm
from seglib import cgp2d 
from seglib.clustering.ce_multicut import *
from seglib.preprocessing import norm01,normCProb,reshapeToImage,normCProbFlat

import pylab





img = "img/37073.jpg"
img = "img/156065.jpg"
img = numpy.squeeze(vigra.readImage(img))#[0:75,0:75,:]
lab = vigra.colors.transform_RGB2Lab(img)
labels ,nseg= vigra.analysis.slicSuperpixels(lab,10.0,25)
labels = vigra.analysis.labelImage(labels).astype(numpy.uint64)
cgp,tgrid = cgp2d.cgpFromLabels(labels)
imgBig = vigra.sampling.resize(lab,cgp.shape)
grad = numpy.squeeze(vigra.filters.gaussianGradientMagnitude(imgBig,2.5))+0.1



#def changeWeights(self,weights,cgp=cgp):
#    self = None
#    self = multicutFromCgp(cgp,weights)
def argDual(self,out):
    arg = self.arg()
    factorSubset=opengm.FactorSubset(gm)
    variableIndices = factorSubset.variableIndices()
    #print variableIndices.shape

    out[:]= arg[variableIndices[:,0]]!=arg[variableIndices[:,1]]

    return out

opengm.inference.Cgc.argDual = argDual
#opengm.inference.Cgc.changeWeights = changeWeights

if False :
    imgBigRgb = vigra.sampling.resize(img,cgp.shape)
    cgp2d.visualize(imgBigRgb,cgp)


#weights = imgToWeight(cgp=cgp,img=grad,gamma=50.5,method='exp')





print "accumulate cell hist"
hist = cgp.accumulateCellHistogram(cellType=2,image=img,binCount=8,sigma=1.5)
hist = hist.reshape([cgp.numCells(2),-1]).astype(numpy.float32)
hist = normCProbFlat(hist)
print hist.shape



print "accumulate cell feat"
feat = cgp.accumulateCellFeatures(cellType=2,image=img,features='Mean')[0]['Mean']
feat = feat.reshape([cgp.numCells(2),-1]).astype(numpy.float32)
#feat = normCProbFlat(hist)
print feat.shape


edge=cgp.cell2ToCell1Feature(feat,mode='l2')
print edge.min(),edge.max()

e1=numpy.exp(-0.00023*edge)
e0=1.0-e1
w=e1-e0






cgc,gm  = multicutFromCgp(cgp=cgp,weights=w)
nFac    = cgp.numCells(1)
nVar    = cgp.numCells(2)


cgc.infer(cgc.verboseVisitor())



argDual     = numpy.ones(nFac,dtype=opengm.label_type)
argDual=cgc.argDual(out=argDual)
argDual=argDual.astype(numpy.float32)
ce = CeMc(cgp=cgp)


if True :
    imgBigRgb = vigra.sampling.resize(img,cgp.shape)
    cgp2d.visualize(imgBigRgb,cgp,edge_data_in=argDual)


initWeights = w.copy()
weightsMean = w.copy()
weightsStd  = numpy.ones(nFac,dtype=numpy.float32)*0.25









#hist=vigra.taggedView(hist,"xc")
#hist=hist.transposeToVigraOrder()


#hist=numpy.array(hist)
print "construkt"
hlo = cgp2d.HighLevelObjective(cgp)

print "set features"
hlo.setRegionFeatures(feat)












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


        return unaryCost#+0.1*sizeCost+0.1*sum3 + 0.1*sum4



obj = Objective(cgp,initWeights)


nSamples = 50
nElites  = 1


mixLow    = 0.01
mixHigh   = 0.5

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
    print  iteration
    for sampleIndex in range(nSamples):
        # sample new weights
        offset  = gaussOffset(0.0,0.007)
        weights = sampleFromGauss(mean=weightsMean,std=weightsStd,out=w) + offset
        # update multicut weights
        #cgc.changeWeights(weights)
        cgc =  multicutFromCgp(cgp,weights)[0]
        # infer
        #print  sampleIndex,iteration
        #cgc.infer(cgc.verboseVisitor())
        cgc.infer()
        argDual[sampleIndex,:]   = cgc.argDual(out=argDual[sampleIndex,:]) 
        argPrimal[sampleIndex,:] = cgc.arg() 
       
    
        #print "evaluate new"
        hlo.setPrimalLabels(argPrimal[sampleIndex,:])
        #@print "merge features"
        hlo.mergeFeatures()

        within  = hlo.withinClusterDistance();
        between = hlo.betweenClusterDistance("squaredNorm",0.05)

        print "remerge",between
        remerged = hlo.writeBackMergedFeatures()
        edgeSum=cgp.cell2ToCell1Feature(remerged,mode='l2')
        """
        for c in range(remerged.shape[-1]):
            print c
            r0=cgp.featureToImage(cellType=2,features=remerged[:,c],ignoreInactive=False,inactiveValue=0.0,useTopologicalShape=False)
            pylab.imshow(    numpy.swapaxes(norm01(r0),0,1) )
            pylab.show()
        """
        #r1=cgp.featureToImage(cellType=2,features=remerged[:,1],ignoreInactive=False,inactiveValue=0.0,useTopologicalShape=False)
        #r2=cgp.featureToImage(cellType=2,features=remerged[:,2],ignoreInactive=False,inactiveValue=0.0,useTopologicalShape=False)

        #imgM=img.copy()

        #imgM[:,:,0]=r0
        #imgM[:,:,1]=r1
        #imgM[:,:,2]=r2



        #


        #pylab.imshow(    numpy.swapaxes(norm01(imgM[:,:,:]),0,1) )
        #pylab.show()

        #imgBigRgb = vigra.sampling.resize(img,cgp.shape)
        #cgp2d.visualize(imgBigRgb,cgp,edge_data_in=edge)




        #print "w",within,"b",between,"w-b",within - between

        #print "b",between*15.0

        addOn    = between#*15.0

        regular  = obj(argPrimal=argPrimal[sampleIndex,:],argDual=argDual[sampleIndex,:])
        #print "reg",regular,"addOn",addOn
        objVal[sampleIndex] =  addOn   #+ regular

        weightSamples[sampleIndex,:] = weights[:]
        weightOffset[sampleIndex] = offset
        #print "objval ",objVal[sampleIndex],"bestObj",bestObj

        if objVal[sampleIndex] < bestObj:
            foundBest=True
            bestObj=objVal[sampleIndex]
            print "bestObj",bestObj 
            bestState=argDual[sampleIndex,:].copy()
            print "meanstate from best",numpy.mean(bestState),"best obj val",objVal[sampleIndex]
            #imgBigRgb = vigra.sampling.resize(img,cgp.shape)
            #cgp2d.visualize(imgBigRgb,cgp,edge_data_in=bestState.astype(numpy.float32))
            
        if iteration%50==0 and sampleIndex==0:
            print "meanstate from best",numpy.mean(bestState),"best obj val",objVal[sampleIndex]
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

    whereSmall = numpy.where(weightsStdNew<0.4)
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