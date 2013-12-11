import vigra
import numpy
import opengm
from seglib import cgp2d 
from seglib.clustering.ce_multicut import *
from seglib.preprocessing import norm01,normCProb,reshapeToImage,normCProbFlat
from seglib.igo import Igo
import pylab
from sklearn.cluster import Ward,WardAgglomeration




img = "img/37073.jpg"
img = "img/156065.jpg"
img = numpy.squeeze(vigra.readImage(img))#[0:75,0:75,:]
lab = vigra.colors.transform_RGB2Lab(img)
labels ,nseg= vigra.analysis.slicSuperpixels(lab,10.0,5)
labels = vigra.analysis.labelImage(labels).astype(numpy.uint64)
cgp,tgrid = cgp2d.cgpFromLabels(labels)
imgBig = vigra.sampling.resize(lab,cgp.shape)
imgBigRGB = vigra.sampling.resize(img,cgp.shape)
grad = numpy.squeeze(vigra.filters.gaussianGradientMagnitude(imgBig,2.5))+0.1




if False :
    imgBigRgb = vigra.sampling.resize(img,cgp.shape)
    cgp2d.visualize(imgBigRgb,cgp)




class CgpClustering(object):
    def __init__(self,cgp):
        self.cgp = cgp 
        self.labels    = numpy.zeros(self.cgp.numCells(2),dtype=numpy.uint64)

class HierarchicalClustering(CgpClustering):
    def __init__(self,cgp):
        super(HierarchicalClustering, self).__init__(cgp)
        self.connectivity   = cgp.sparseAdjacencyMatrix()

    def segment(self,features,nClusters):

        #print "features",features.shape
        #print "self.connectivity",self.connectivity.shape

        self.ward = WardAgglomeration(n_clusters=nClusters, connectivity=self.connectivity).fit(features.T)
        self.labels[:] = self.ward.labels_

    def mergedCgp(self):

        newLabels  = self.cgp.featureToImage(cellType=2,features=self.labels.astype(numpy.float32),useTopologicalShape=False)
        cgp,tgrid = cgp2d.cgpFromLabels(newLabels.astype(numpy.uint64)+1)
        return cgp,tgrid


print "accumulate cell feat"
feat = cgp.accumulateCellFeatures(cellType=2,image=lab,features='Mean')[0]['Mean']
feat = feat.reshape([cgp.numCells(2),-1]).astype(numpy.float32)


hc = HierarchicalClustering(cgp=cgp)
hc.segment(feat,100)
mcgp,mtgrid = hc.mergedCgp()

cgp = mcgp


weights = imgToWeight(cgp=cgp,img=grad,gamma=50.5,method='exp')


if False :
    cgp2d.visualize(imgBigRGB,cgp)


class Objective(object):
    def __init__(self,cgp,weights,feat=None):
        self.cgp     = cgp
        self.weights = weights
        self.initWeights = weights.copy()
        self.cgc,self.gm  = multicutFromCgp(cgp=cgp,weights=self.weights)
        self.feat =feat
        print "construkt"
        #self.hlo = cgp2d.HighLevelObjective(self.cgp)

        print "set features"
        #hlo.setRegionFeatures(self.feat)


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
    def numVar(self):
        return self.cgp.numCells(1)

    def show(self,weights):
        self.cgc.changeWeights(weights)
        self.cgc.infer()
        argDual=self.cgc.argDual()

        cgp2d.visualize(imgBigRGB,cgp=self.cgp,edge_data_in=argDual.astype(numpy.float32))

    def evaluate(self,weights):
        self.weights=weights.copy()
        #self.weights/=self.weights.max()
        self.cgc.changeWeights(weights)
        self.cgc.infer()
        argDual=self.cgc.argDual()
        argPrimal=self.cgc.arg()
        where1    = numpy.where(argDual==1)
        unaryCost = numpy.sum(self.initWeights[where1])



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


        return unaryCost+0.1*sizeCost+5.0*sum3 + 20.0*sum4




objective = Objective(cgp=cgp,weights=weights)
igo=Igo(objective=objective)
igo.setStartingPoint(mean=weights,covariance=numpy.eye(objective.numVar())*2.0)
igo.infer()