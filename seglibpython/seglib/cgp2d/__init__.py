from _cgp2d import *
import numpy
import numpy.linalg
import vigra
from scipy.sparse import coo_matrix
import scipy
#from featurecalc import emd as earthmd

import pylab
from scipy import sparse






def regionAffinity(labelings,out=None):
  nRegion    = labelings.shape[1]
  nLabelings = labelings.shape[1]

  nLabels    = numpy.max(labelings,axis=1)

  l  = numpy.require(labelings,dtype=numpy.uint64)
  nl = numpy.require(labelings,dtype=numpy.uint64)

  return _regionAffinity(l,nl,out)





def affinityDiffusion(W,tMax,sparse=True,visitor=None):
    n = W.shape[0]

    
    
    if sparse:
        W0=sparse.coo_matrix(W)
        # setup d
        diagonals = [[1,2,3,4], [1,2,3], [1,2]]
        D         = diags([numpy.sum(W,axis=1)], [0])
        invertedD = scipy.sparse.linalg.inv(D)

        # compute smoothing kernel
        P = invertedD.dot(W0)

        # eye
        I = sparse.eye(n)

        for t in range(tMax):
            W1 = W0.dot(P) + I
            W0 = W1.copy()

        return W1.todense()*invertedD.todense()
    else :
        # set up D
        D = numpy.zeros([n,n])
        numpy.fill_diagonal(D,numpy.sum(W,axis=1))

        invertedD = numpy.linalg.inv(D)

        # compute smoothing kernel
        P = numpy.dot(invertedD , W )

        # iterate
        W0 = W.copy()
        W1 = W.copy()

        
        I =  numpy.eye(n)
        for t in range(tMax):
            W1[:,:] = numpy.dot(W0,P) + I
            W0[:,:] = W1[:,:]

            if visitor is not None:
                exit = visitor(W1)


        return W1*invertedD



def gaussianSmoothing1d(histograms,sigma):
  nHist = histograms.shape[0]
  nBins = histograms.shape[1]

  kernel=vigra.filters.Kernel1D()
  kernel.initDiscreteGaussian(sigma)

  smoothedHistograms=vigra.filters.convolveOneDimension(histograms, dim=1, kernel=kernel)

  return smoothedHistograms





class Region2Boundary(object):
  @staticmethod
  def _normDiff(a,b,norm):
      assert a.shape == b.shape
      result = numpy.sum(numpy.abs(a-b)**norm,axis=1)
      assert result.shape[0]==a.shape[0]
      return result

  @staticmethod
  def l1(a,b):
    return Region2Boundary._normDiff(a,b,1)

  @staticmethod
  def l2(a,b):
    return Region2Boundary._normDiff(a,b,2)

  @staticmethod
  def logL1(a,b):
    assert a.min()>=0
    assert b.min()>=0
    return Region2Boundary._normDiff(numpy.log(a+1),numpy.log(b+1),1)

  @staticmethod
  def logL2(a,b):
    assert a.min()>=0
    assert b.min()>=0
    return Region2Boundary._normDiff(numpy.log(a+1),numpy.log(b+1),2)

  @staticmethod
  def chi2(a,b):
    assert a.shape == b.shape
    assert a.ndim == 2


    nItems = a.shape[0]
    nBins  = a.shape[1]
    binWise = numpy.zeros([nItems,nBins],dtype=a.dtype)
    for x in xrange(nBins):
        P_i  = a[:,x]
        Q_i  = b[:,x]
        PQ_sdiff = ( P_i - Q_i ) ** 2 
        PQ_sum   = ( P_i + Q_i )

        whereNotZero = numpy.where(PQ_sum!=0)
        whereZero    = numpy.where(PQ_sum==0)

        binWise[whereNotZero[0],x]=PQ_sdiff[whereNotZero]/PQ_sum[whereNotZero]

    xDiff = numpy.sum(binWise,axis=1)*0.5
    assert xDiff.shape[0]==nItems
    return xDiff

  @staticmethod
  def emd(a,b):
    ret = numpy.zeros(a.shape[0])
    for i in range(a.shape[0]):
      ret[i] = earthmd(a[i,:], b[i,:])    
    return ret

  @staticmethod
  def logChi2(a,b):
    assert a.min()>=0
    assert b.min()>=0
    return Region2Boundary.chi2(numpy.log(a+1),numpy.log(b+1))




def cgpFromLabels(labels):
    tgrid=TopologicalGrid(labels)
    cgp=Cgp(tgrid)
    return cgp ,tgrid
  
metaclass_more_cgp = Cgp.__class__ 
metaclass_more_cell0 = Cell0.__class__ 
metaclass_more_cell1 = Cell1.__class__ 
metaclass_more_cell2 = Cell2.__class__ 

class injector_more_cgp(object):
    class __metaclass__(metaclass_more_cgp):
        def __init__(self, name, bases, dict):
            for b in bases:
                if type(b) not in (self, type):
                    for k,v in dict.items():
                        setattr(b,k,v)
            return type.__init__(self, name, bases, dict)

class injector_more_cell0(object):
    class __metaclass__(metaclass_more_cell0):
        def __init__(self, name, bases, dict):
            for b in bases:
                if type(b) not in (self, type):
                    for k,v in dict.items():
                        setattr(b,k,v)
            return type.__init__(self, name, bases, dict)

class injector_more_cell1(object):
    class __metaclass__(metaclass_more_cell1):
        def __init__(self, name, bases, dict):
            for b in bases:
                if type(b) not in (self, type):
                    for k,v in dict.items():
                        setattr(b,k,v)
            return type.__init__(self, name, bases, dict)

class injector_more_cell2(object):
    class __metaclass__(metaclass_more_cell2):
        def __init__(self, name, bases, dict):
            for b in bases:
                if type(b) not in (self, type):
                    for k,v in dict.items():
                        setattr(b,k,v)
            return type.__init__(self, name, bases, dict)


def spatialShapeMatch(shapeA,shapeB):
  return shapeA[0]==shapeB[0] and shapeA[1]==shapeB[1]




class WeightTransformation(object):
  
  @staticmethod 
  def exp(weights,gamma):
    assert gamma < 0
    return numpy.exp(gamma*weights)

  @staticmethod 
  def raw(weights):
    return weights.copy()

  @staticmethod 
  def clipQuantiles(weights,ql=0.05,qh=0.95):
    raise RuntimeError("not implemented")


  @staticmethod
  def normalize01(weights):
    w=weights.copy()
    wmin=w.min()
    wmax=w.max()
    w-=wmin
    w/=(wmax-wmin)
    return w



class Cell1WeightTransformation(object):


    @staticmethod
    def ucm(cgp,weights,accType='median'):
      raise RuntimeError("not implemented")



    @staticmethod
    def multicutStabelizer(cgp,weights,ql=0.1,qh=0.9,steps=20):
      raise RuntimeError(" not implemented")

      


    @staticmethod 
    def realativeWeight(cgp,weights,accType='mean',runs=1):
      assert weights.ndim == 1
      assert weights.shape[0] == cgp.numCells(1)
      w=numpy.require(weights,dtype=numpy.float32).copy()
      tWeights=Cell1WeightTransformation.adjacencyStats(cgp,w,accType=accType,runs=runs)
      whereNonZero = numpy.where(tWeights!=0)
      result = numpy.ones(cgp.numCells(1),dtype=numpy.float32)
      result[whereNonZero] = w[whereNonZero]/tWeights[whereNonZero]
      return result
    @staticmethod 
    def diffWeight(cgp,weights,accType='mean',runs=1):
      assert weights.ndim == 1
      assert weights.shape[0] == cgp.numCells(1)
      w=numpy.require(weights,dtype=numpy.float32)
      tWeights=Cell1WeightTransformation.adjacencyStats(cgp,w,accType=accType,runs=runs)
      result = w-tWeights



      return result

    @staticmethod
    def bilateralMean(cgp,weights,alpha,gamma,runs=1):
        assert weights.ndim == 1
        assert weights.shape[0] == cgp.numCells(1)

        assert alpha >  0.0
        assert alpha <  1.0
        assert gamma >  0.0
        assert runs  >= 1

        w=numpy.require(weights,dtype=numpy.float32).copy()
        for r in range(runs):
            wRes = cgp._cell1GraphBiMean(w,alpha,gamma)
            w[:]=wRes[:]
        return wRes
    @staticmethod
    def stealing(cgp,weights,fraction=0.5,ql=0.1,qh=0.5,runs=1):
        assert weights.ndim == 1
        assert weights.shape[0] == cgp.numCells(1)

        w=numpy.require(weights,dtype=numpy.float32).copy()
        for r in range(runs):
            wRes = cgp._cell1GraphStealing(w,float(fraction),ql,qh)
            w[:]=wRes[:]
        return wRes

    @staticmethod
    def stealFromWeakest(cgp,weights,runs=1):
        assert weights.ndim == 1
        assert weights.shape[0] == cgp.numCells(1)

        w=numpy.require(weights,dtype=numpy.float32).copy()
        for r in range(runs):
            wRes = cgp._cell1GraphPropagation(w)
            w[:]=wRes[:]
        return wRes

    @staticmethod
    def adjacencyStats(cgp,weights,accType,runs=1):
        assert weights.ndim == 1
        assert weights.shape[0] == cgp.numCells(1)

        w=numpy.require(weights,dtype=numpy.float32).copy()
        for r in range(runs):
            wRes = Cell1WeightTransformation._adjacencyStats(cgp,w,accType)
            w[:]=wRes[:]
        return wRes


    @staticmethod
    def _adjacencyStats(cgp,weights,accType):
        assert weights.ndim == 1
        assert weights.shape[0] == cgp.numCells(1)
        accTypeDict ={ 'min':0,'max':1,'mean':2,'median':3}
        assert accType in accTypeDict

        resultWeights = cgp._cell1GraphAdjAcc(weights)
        result = resultWeights[:,accTypeDict[accType]].copy()
        resultWeights = None
        return result



class Cell1Features(object):
  @staticmethod
  def boarderTouch(cgp):
    return cgp._cell1BoarderTouch()

  @staticmethod
  def countMultiEdges(cgp):
    return cgp._cell1countMultiEdges()

  @staticmethod
  def relativeCenterDist(cgp):
    return cgp._cell1RelativeCenterDist()

  @staticmethod
  def geometricFeatures(cgp):
    names = [
      'lineSize','bBoxDiagonal','bBoxDiagonal/lineSize','startEndDist','startEndDist/lineSize',
      'adjRegSizeMean','adjRegSizeAbsDiff','adjRegSizeMin','adjRegSizeMax',
      'adjRegRelSizeMean','adjRegRelSizeAbsDiff','adjRegRelSizeMin','adjRegRelSizeMax'
    ]
    features = cgp._cell1GeoFeatures()
    assert features.shape[1] == len(names)
    return features,names

  @staticmethod
  def topologicalFeatures(cgp):
    names = [
      'nCell1Adj','adjRegNCell2AdjMean','adjRegNCell2AdjAbsDiff','adjRegNCell2AdjMin','adjRegNCell2AdjMax'
    ]
    features = cgp._cell1TopoFeatures()
    assert features.shape[1] == len(names)
    return features,names





class more_cgp(injector_more_cgp, Cgp):
    def _orientedWatershedTransform(self,pixelWiseAngles):
      print pixelWiseAngles.shape
      return self.owt(numpy.require(pixelWiseAngles,dtype=numpy.float32))



    def sparseAdjacencyMatrix(self):
      cell1Bounds=self.cell1BoundsArray()-1


      def unique_rows(a):
        a = numpy.ascontiguousarray(a)
        unique_a = numpy.unique(a.view([('', a.dtype)]*a.shape[1]))
        return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

      cell1Bounds = unique_rows(cell1Bounds.T).T

      row  = numpy.hstack([ cell1Bounds[:,0], cell1Bounds[:,1] ,numpy.arange(self.numCells(2))])
      col  = numpy.hstack([ cell1Bounds[:,1], cell1Bounds[:,0] ,numpy.arange(self.numCells(2))])

      



      #print ""

      data = numpy.ones(len(col))
      cm =  coo_matrix((data, (row,col)), shape=[self.numCells(2)]*2 )
      cm = scipy.sparse.csr_matrix(cm)

      return cm

    def cell2ToCell1Feature(self,regionFeatures,mode='chi2',dtype=numpy.float32):
      mode = mode.lower()
      regionFeatures = regionFeatures.reshape([regionFeatures.shape[0],-1])

      assert mode in ['logchi2','chi2','l1','logl1','l2','logl2', 'emd']
      assert regionFeatures.shape[0]==self.numCells(2)

      cell1Bounds=self.cell1BoundsArray()
      assert cell1Bounds.min()==1
      assert cell1Bounds.max()==self.numCells(2)
      assert cell1Bounds.shape[0]==self.numCells(1)
      assert cell1Bounds.shape[1]==2
      r0 = regionFeatures[ cell1Bounds[:,0]-1,:]
      r1 = regionFeatures[ cell1Bounds[:,1]-1,:]

      if  mode == 'chi2':
        return Region2Boundary.chi2(r0,r1)
      elif mode == 'logchi2':
        return Region2Boundary.logChi2(r0,r1)
      elif mode == 'l1':
        return Region2Boundary.l1(r0,r1)
      elif mode == 'logl21':
        return Region2Boundary.logL1(r0,r1)
      elif mode == 'l2':
        return Region2Boundary.l2(r0,r1)
      elif mode == 'logl2':
        return Region2Boundary.logL2(r0,r1)
      elif mode == 'emd':
        return Region2Boundary.emd(r0,r1)

    def labelGrid(self,cellType,useTopologicalShape=True,out=None):
      lgrid=self.tgrid.labelGrid(cellType=cellType,useTopologicalShape=useTopologicalShape,out=out)
      if (cellType !=2):
        assert lgrid.min()==0
      assert lgrid.max()==self.numCells(cellType)
      return lgrid


    def pixelHistToCell2Hist(self,pixelHist):
        """  calculate a histogram for each region based
        on a pixel wise histogram 
        """  

        shape = self.shapeLabeling
        nBins = pixelHist.shape[2]
        assert shape[0]==pixelHist.shape[0]
        assert shape[1]==pixelHist.shape[1]

        labels=self.labelGrid(cellType=2,useTopologicalShape=False)

        #print "shapeLabeling",shape,";labelsShape ",labels.shape

        assert shape[0]==labels.shape[0]
        assert shape[1]==labels.shape[1]
        assert labels.min()==1
        assert labels.max()==self.numCells(2)

        inputData = numpy.require(pixelHist,dtype=numpy.float32)
        featureDict = vigra.analysis.extractRegionFeatures(image=inputData,labels=labels,features='mean',ignoreLabel=0)
        regionHist  =featureDict['mean']
        assert regionHist.ndim == 2
        assert regionHist.shape[0]==self.numCells(2)+1
        assert regionHist.shape[1]==nBins
        returnArray = regionHist[1:self.numCells(2)+1,:].copy()
        assert returnArray.shape[0]==self.numCells(2)

        regionHist=None
        featureDict=None
        
        return returnArray

    
    def accumulateCellHistogram(self,cellType,image,histogramRange=None,binCount=64,useCellMinMax=False,sigma=None):
      assert image.ndim ==2 or image.ndim==3

      data=image.reshape([image.shape[0],image.shape[1],-1])
      nChannels = data.shape[2]
      nCells    = self.numCells(cellType)

      # result array 
      cellHisto = numpy.ones([nCells,nChannels,binCount],dtype=numpy.float32)


      if histogramRange is not None:
        histogramRange=numpy.require(histogramRange,dtype=numpy.float32)
        assert histogramRange.ndim==2
        assert histogramRange.shape[0]==nChannels
        assert histogramRange.shape[1]==2


      # iterate over every channel
      for c in range(nChannels):

        # get histogram range and channel of data
        if histogramRange is None:
          hr = None 
        else :
          hr = histogramRange[c,:]
        d = data[:,:,c]

        # accumulate histogram for one(!) channel
        featureDict,activeFeatures = self.accumulateCellFeatures( cellType=cellType,image=d,features=['Histogram','Count'],
                                                                  histogramRange=hr, binCount=binCount,
                                                                  useCellMinMax=useCellMinMax,histMode=True)
        # write channel result into result array
        assert 'Histogram' in activeFeatures

        #print activeFeatures

        channelHist = featureDict['Histogram']
        channelHistCount = featureDict['Count']

        channelHistCount2=numpy.repeat(channelHistCount, binCount)
        channelHistCount2=channelHistCount2.reshape(-1,binCount)

        #print "chshape",channelHist.shape
        #print "cpunt ",channelHistCount2.shape


        #channelHistCount = 
        #print channelHist.reshape(01,channelHistCount.shape

        #channelHist=channelHist.reshape(-1)
        channelHist/=channelHistCount2



        #channelHist=channelHist.reshape([-1,binCount])
        #print "chshape",channelHist.shape
        assert channelHist.ndim == 2
        assert channelHist.shape[0]==nCells
        assert channelHist.shape[1]==binCount

        cellHisto[:,c,:]=channelHist[:,:]

      if sigma is not None:
        cellHisto2d = cellHisto.reshape([-1,binCount])
        cellHisto2d = gaussianSmoothing1d(cellHisto2d,sigma)
        cellHisto   = cellHisto2d.reshape([nCells,nChannels,binCount])

      return cellHisto








    def accumulateCellFeatures(self,cellType,image,features='all',histogramRange=None,binCount=64,useCellMinMax=False,histMode=False):

      # check for valid input
      dataShape = image.shape
      if spatialShapeMatch(dataShape,self.shape):
        useTopologicalShape=True
      elif spatialShapeMatch(dataShape,self.shapeLabeling):
        useTopologicalShape=False
      else :
        raise RuntimeError("image.shape does neither match cgp.shape nor cgp.shapeLabeling")

      image = numpy.require(image,dtype=numpy.float32)

      nCells=self.numCells(cellType)
      #labels=self.labelGrid(cellType)
      #labels=numpy.ones(self.shape,dtype=numpy.uint32)
      labels=self.labelGrid(cellType=cellType,useTopologicalShape=useTopologicalShape)
      

      if histMode :
        hFeatures = ['Histogram','Count']
        assert image.ndim == 2
        if histogramRange  is None :
          if(useCellMinMax==False):
            histogramRange=(float(image.min()),float(image.max()))
          else:
            histogramRange='globalminmax'
        else:
          histogramRange = (float(histogramRange[0]),float(histogramRange[1]))
        values=vigra.analysis.extractRegionFeatures(image=image ,labels=labels, features=hFeatures, histogramRange=histogramRange ,binCount=binCount) 

      else:
        values=vigra.analysis.extractRegionFeatures(image=image,labels=labels,features=features,ignoreLabel=0)
      activeFeatures=values.activeFeatures()
      #del values
      
      featureDict=dict()
      for fname in activeFeatures :
        featureVals=values[fname]
        if isinstance(featureVals, numpy.ndarray) or issubclass(featureVals.__class__,numpy.ndarray):
          shape=featureVals.shape
          dim=len(shape)
          if dim==1:
            featureDict[fname]=featureVals[1:nCells+1].copy()
          elif dim==2:
            featureDict[fname]=featureVals[1:nCells+1,:].copy()
          elif dim==3:
            featureDict[fname]=featureVals[1:nCells+1,:,:].copy()
        elif isinstance(featureVals,(int ,long,float)):
          featureDict[fname]=featureVals
        else :
          raise RuntimeError("internal error in accumulateCellFeatures")

      values=None
      return featureDict,activeFeatures
    
    def cells(self,cellType):
      if(cellType==0):
        return self.cells0
      elif(cellType==1):
        return self.cells1
      elif(cellType==2):
        return self.cells2
      else:
        raise NameError("cellType must be 0,1,or 2")

    def matchMergedCgpCells(self,coarse_cgp):
      # fine cells to coarse label(s)
      cell_to_coarse=[dict() ,dict(), dict() ]
      # coarse labels to fine cell labels
      cell_to_fine=[None]*3
      cell_to_fine[0]=[None]*coarse_cgp.numCells(0)
      cell_to_fine[1]=[ list() ]*coarse_cgp.numCells(1)
      cell_to_fine[2]=[ list() ]*coarse_cgp.numCells(2)

      coarseLabeling=numpy.ones(self.shape,dtype=numpy.uint32)
      for cellType in range(3):
        coarseLabeling=coarse_cgp.labelGrid(cellType,out=coarseLabeling)
        for cell in self.cells(cellType):
          label=cell.label
          aPoint=cell.points[0]
          coarseLabel=coarseLabeling(aPoint)
          if coarseLabel!=0:
            # cell is still active in coarse graph
            cell_to_coarse[cellType][ label - 1 ]=coarseLabel
            if cellType!=0 :
              cell_to_fine[cellType][coarseLabel-1].append(label)
            else:
              cell_to_fine[cellType][coarseLabel-1]=label

      return cell_to_coarse,cell_to_fine






class _cell_helper(object):
  
  @staticmethod
  def adjacencyGen(cell):

    cgp=cell.cgp
    cellType=cell.cellType
    # get own cell label
    cellLabel=cell.label
    #get cells below
    assert cellType!=0
    cellsBelow=cgp.cells(cellType-1)

    for boundedByCellLabel in cell.boundedBy:
      # index of junction
      boundedByCellIndex=boundedByCellLabel-1
      # get bounds of boundedByCell
      bounds = cellsBelow[boundedByCellIndex].bounds
      for otherCellLabel in bounds:
        if otherCellLabel != cellLabel:
          yield otherCellLabel , boundedByCellLabel

  @staticmethod
  def adjacentCellsGen(cell):
    cells=cell.cgp.cells(cell.cellType)
    for cellLabel in _cell_helper.adjacencyGen(cell):
      yield cells[cellLabel-1]

  @staticmethod
  def boundingCellsGen(cell):
    assert cell.cellType <=1
    higherCells=cell.cgp.cells(cell.cellType+1)
    for label in cell.bounds:
      yield higherCells[label-1]

  @staticmethod
  def boundingByCellsGen(cell):
    assert cell.cellType >=1
    lowerCells=cell.cgp.cells(cell.cellType-1)
    for label in cell.boundedBy:
      yield lowerCells[label-1]


class more_cell0(injector_more_cell0,Cell0):

  def boundingCellsGen(self):
    return _cell_helper.boundingCellsGen(self)

class more_cell1(injector_more_cell1,Cell1):

  def adjacencyGen(self):
    return _cell_helper.adjacencyGen(self)
  def adjacentCellsGen(self):
    return _cell_helper.adjacentCellsGen(self)
  def boundingCellsGen(self):
    return _cell_helper.boundingCellsGen(self)
  def boundedByCellsGen(self):
    return _cell_helper.boundedByCellsGen(self)

class more_cell2(injector_more_cell2,Cell2):

  def adjacencyGen(self):
    return _cell_helper.adjacencyGen(self)
  def adjacentCellsGen(self):
    return _cell_helper.adjacentCellsGen(self)
  def boundedByCellsGen(self):
    return _cell_helper.boundedByCellsGen(self)







def shortest_path(cgp, sourceNode, edgeWeight):
    """   
    @attention All weights must be nonnegative.

    @type  graph: graph
    @param graph: Graph.

    @type  sourceNode: node
    @param sourceNode: Node from which to start the search.

    @rtype  tuple
    @return A tuple containing two dictionaries, each keyed by
        targetNodes.  The first dictionary provides the shortest distance
        from the sourceNode to the targetNode.  The second dictionary
        provides the previous node in the shortest path traversal.
        Inaccessible targetNodes do not appear in either dictionary.
    """
    # Initialization
    dist     = { sourceNode: 0 }
    previous = {}

    q = set( cellIndex+1 for cellIndex in xrange(cgp.numCells(1) ) )

    #q = graph.get_nodes()

    cells1=cgp.cells1

    # Algorithm loop
    counter=0
    while q:
        # examine_min process performed using O(nodes) pass here.
        # May be improved using another examine_min data structure.
        # See http://www.personal.kent.edu/~rmuhamma/Algorithms/MyAlgorithms/GraphAlgor/dijkstraAlgor.htm
        #u = q[0]

        u = iter(q).next()

        for cellLabel in q:
            if (   (not dist.has_key(u))  or (dist.has_key(cellLabel) and dist[cellLabel] < dist[u]) ):
                u = cellLabel
        q.remove(u)
        if counter%50 ==0 :
          print "c=",counter," u=",u
        counter+=1
        # Process reachable, remaining nodes from u

        for v ,connector in cells1[u-1].adjacencyGen():
            if v in q:
                #alt = dist[u] + graph.get_arrow_weight(u, v)

                alt=  dist[u] + edgeWeight[v-1]
                if (not dist.has_key(v)) or (alt < dist[v]):
                    dist[v] = alt
                    previous[v] = u,connector

    return (dist, previous)

#######

#from reducer import *
#from filterbank import *
#from oversegmentation import *

import numpy
import glob
import os
import sys
import vigra
from  matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def getCgpPointSet(cgp,cellType=2):
    pointSetList=[None]*len(cgp.cells(cellType)) 
    for i,cell in enumerate(cgp.cells(cellType)):
        pointSetList[i]=set()
        points=cell.points
        for p in points:
            pointSetList[i].add(p)

    return pointSetList




def visualize(
    img_rgb,
    cgp,
    edge_data_in=None,
    show=True,
    cmap=cm.jet,
    title=None
):
    img_rgb_raw=img_rgb.copy()
    if edge_data_in is not None:
        edge_data=edge_data_in.copy()
    else:
        edge_data=None
    img_rgb=numpy.squeeze(img_rgb)
    if img_rgb.ndim == 2:
      img = numpy.ones([img_rgb.shape[0],img_rgb.shape[1],3 ])
      img[:,:,0] = img_rgb[:,:]
      img[:,:,1] = img_rgb[:,:]
      img[:,:,2] = img_rgb[:,:]
    else :
      img=img_rgb.copy()
    img-=img.min()
    img/=img.max()
    # get edge map
    edgeMarkers=cgp.labelGrid(1,True)
    whereEdges=numpy.where(edgeMarkers!=0)
    edgeMarkers[whereEdges]=1

    


    if edge_data is not None :

        #edge_data=numpy.sqrt(edge_data)

        resMin=numpy.min(edge_data)
        resMax=numpy.max(edge_data)

        #print "mi ma",resMin,resMax
        edge_data[:]=(edge_data[:]-resMin)/(resMax-resMin)


        resImg=cgp.featureToImage(cellType=1,features=edge_data,ignoreInactive=False,inactiveValue=0.0)


        edgeValues=resImg[whereEdges]

        # transform 
        mycm=cm.ScalarMappable(norm=None, cmap=cmap)
        mycm.set_array(edgeValues.reshape(-1))

        colorArray=mycm.to_rgba(edgeValues)

        #print " shape ",colorArray.shape
        #print colorArray
        #img*=255
        img[whereEdges[0],whereEdges[1],0]=colorArray[:,0]
        img[whereEdges[0],whereEdges[1],1]=colorArray[:,1]
        img[whereEdges[0],whereEdges[1],2]=colorArray[:,2]
        """
        img[whereEdges[0],whereEdges[1],0]=(1.0-resImg[whereEdges[0],whereEdges[1]])
        img[whereEdges[0],whereEdges[1],1]=0.0#resImg[whereEdges[0],whereEdges[1]]*255.0
        img[whereEdges[0],whereEdges[1],2]=resImg[whereEdges[0],whereEdges[1]]
        """

    elif edge_data is None :
      labelImage=cgp.tgrid.labelGrid(2,False)

      cedge     = vigra.analysis.regionImageToCrackEdgeImage(numpy.require(labelImage,dtype=numpy.uint32))

      #cedge[cedge!=0]=0
      whereEdges=numpy.where(cedge==0)

      #img/=255
      img[whereEdges[0],whereEdges[1],0]=0.0
      img[whereEdges[0],whereEdges[1],1]=0.0
      img[whereEdges[0],whereEdges[1],2]=0.0

    else :
        #img#/=255
        #img[whereEdges[0],whereEdges[1],0]=0.0
        #img[whereEdges[0],whereEdges[1],1]=0.0
        #img[whereEdges[0],whereEdges[1],2]=0.0
        #edgeData=numpy.ones()
        #resImg=cgp.featureToImage(cellType=1,features=whereEdges.astype(numpy.float32),ignoreInactive=False,inactiveValue=0.0)
        resImg=vigra.filters.discDilation(edgeMarkers.astype(numpy.uint8),1)

        whereEdges=numpy.where(resImg!=0)

        img[whereEdges[0],whereEdges[1],0]=0.0
        img[whereEdges[0],whereEdges[1],1]=0.0
        img[whereEdges[0],whereEdges[1],2]=0.0



    f = pylab.figure()
    for n, iimg in enumerate([img,img_rgb_raw/255]):
        #f.add_subplot(2, 1, n)  # this line outputs images on top of each other
        f.add_subplot(1, 2, n)  # this line outputs images side-by-side
        plt.imshow(numpy.swapaxes(iimg,0,1))



    #plt.imshow(  numpy.flipud(numpy.rot90(img) )  ,interpolation=None)
    if title is not None:
      plt.title(title)
    if(show):
        plt.show()



def loadSegGtFile(segfile):
    f = open(segfile, 'r')
    lines= f.readlines()
    i=0
    start=0
    for line in lines:
        if line.startswith("width"):
            width=[int(s) for s in line.split() if s.isdigit()][0]
            #print "width ", width
        if line.startswith("height"):
            height=[int(s) for s in line.split() if s.isdigit()][0]
            #print "height ", height    

        
        if line.startswith("data"):
            start=i+1
            break
        i+=1
    seg=numpy.ones([width,height],dtype=numpy.uint32)
    #seg[:,:]=-1
    for line in lines[start:len(lines)]:
        #print line
        [label,row,cstart,cend]=[int(s) for s in line.split() if s.isdigit()]
        assert (cend +1 <= width)
        assert (row <= height)
        seg[cstart:cend+1,row]=label
    return seg 



def getWideGt(cgp, discDilationRadius=15):
    # get GT as grid
    labelGridGT=cgp.labelGrid(1)
    labelGridGT[labelGridGT!=0]=1
    labelGridGT=labelGridGT.astype(numpy.uint8)
    # make GT wider
    wideGT=vigra.filters.discDilation(labelGridGT,radius=discDilationRadius)
    return wideGT





