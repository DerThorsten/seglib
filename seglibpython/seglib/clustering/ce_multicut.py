from seglib import cgp2d 
from seglib.preprocessing import norm01
import opengm
import numpy
import vigra
from sklearn.cluster import Ward,WardAgglomeration

class CgpClustering(object):
	def __init__(self,cgp):
		self.cgp = cgp 
		self.labels    = numpy.zeros(self.cgp.numCells(2),dtype=numpy.uint64)

class HierarchicalClustering(CgpClustering):
	def __init__(self,cgp):
		super(HierarchicalClustering, self).__init__(cgp)
		self.connectivity 	= cgp.sparseAdjacencyMatrix()

	def segment(self,features,nClusters):

		#print "features",features.shape
		#print "self.connectivity",self.connectivity.shape

		self.ward = WardAgglomeration(n_clusters=nClusters, connectivity=self.connectivity).fit(features.T)
		self.labels[:] = self.ward.labels_

	def mergedCgp(self):

		newLabels  = self.cgp.featureToImage(cellType=2,features=self.labels.astype(numpy.float32),useTopologicalShape=False)
		cgp,tgrid = cgp2d.cgpFromLabels(newLabels.astype(numpy.uint64)+1)
		return cgp,tgrid



def probabilityToWeights(p1,out,beta=0.5):
    assert len(out)==len(p1)
    p0 = 1.0 - p1
    out[:]=numpy.log( p0 / p1 ) + numpy.log((1.0-beta)/beta)
    return out


def sampleFromGauss(mean,std,out):
    #print "mean",mean.shape
    #print "std",std.shape
    #print "out",out.shape
    assert len(mean)==len(std)
    assert len(out)==len(mean)
    n = len(mean)
    samples  = numpy.random.standard_normal(n)
    samples *=std
    samples +=mean
    return samples

def gaussOffset(mean,std):
	return std*float(numpy.random.standard_normal(1))+mean


def gradientToWeight(gradient,gamma):
	#normGrad = norm01(gradient)
	e = numpy.exp(-gamma*gradient)

	e1  = e
	e0  = 1.0-e1

	"""
	print "g ",gradient[:5]
	print "e0",e0[:5]
	print "e1",e1[:5]
	print "w ",(e0-e1)[:5]
	"""
	return e1-e0




def imgToWeight(cgp,img,gamma,method='exp'):
	if tuple(cgp.shape)!=(img.shape):
		img=vigra.sampling.resize(img,cgp.shape)

	img =norm01(img)+0.1  
	img/=1.1

	accgrad = cgp.accumulateCellFeatures(cellType=1,image=img,features='Mean')[0]['Mean']

	if method =='exp':
		weights = gradientToWeight(gradient=accgrad,gamma=gamma)
		return weights
	else :
		raise RuntimeError("not impl")







def multicutFromCgp(cgp,weights=None,parameter=None):
	boundArray 	= cgp.cell1BoundsArray()-1
	nVar 		= cgp.numCells(2)
	nFac 		= cgp.numCells(1)
	space 		= numpy.ones(nVar,dtype=opengm.label_type)*nVar
	gm 			= opengm.gm(space)
	wZero  = numpy.zeros(nFac,dtype=opengm.value_type)
	if weights is None:
		pf=opengm.pottsFunctions([nVar,nVar],wZero,wZero)
	else :
		w = numpy.require(weights,dtype=opengm.value_type)
		pf=opengm.pottsFunctions([nVar,nVar],wZero,w)
	fids = gm.addFunctions(pf)
	gm.addFactors(fids,boundArray)
	cgc = opengm.inference.Cgc(gm=gm,parameter=parameter)



	return cgc,gm



def multicutFromCgp2(cgp,e0,e1,parameter=None):
	boundArray 	= cgp.cell1BoundsArray()-1
	nVar 		= cgp.numCells(2)
	nFac 		= cgp.numCells(1)
	space 		= numpy.ones(nVar,dtype=opengm.label_type)*nVar
	gm 			= opengm.gm(space)
	#w = numpy.require(weights,dtype=opengm.value_type)
	pf=opengm.pottsFunctions([nVar,nVar],e0,e1)

	fids = gm.addFunctions(pf)
	gm.addFactors(fids,boundArray)
	cgc = opengm.inference.Cgc(gm=gm,parameter=parameter)



	return cgc,gm





class AggloCut(object):
	def __init__(self,initCgp,edgeImage,featureImage,rgbImage,siftImage,histImage):
		self.initCgp   		= initCgp
		self.edgeImage 		= edgeImage
		self.featureImage 	= featureImage
		self.rgbImage  		= rgbImage
		self.siftImage 		= siftImage
		self.histImage		= histImage
		#
		self.iterCgp   = initCgp

	def infer(self,gammas,deleteN):
		cgp2d.visualize(self.rgbImage,cgp=self.iterCgp)

		for gamma in gammas:

			# get the weights for this gamma
			#weights = gradientToWeight(self.edgeImage,gamma)
			

	
			#w=e1-e0
			cuts=True
			while(True):
				edge = self.iterCgp.accumulateCellFeatures(cellType=1,image=self.edgeImage,features='Mean')[0]['Mean']
				feat = self.iterCgp.accumulateCellFeatures(cellType=2,image=self.featureImage,features='Mean')[0]['Mean']
				sift = self.iterCgp.accumulateCellFeatures(cellType=2,image=self.siftImage,features='Mean')[0]['Mean']
				hist = self.iterCgp.accumulateCellFeatures(cellType=2,image=self.histImage,features='Mean')[0]['Mean']
				featDiff = numpy.sqrt(self.iterCgp.cell2ToCell1Feature(feat,mode='l2'))/10.0
				siftDiff = (self.iterCgp.cell2ToCell1Feature(sift,mode='chi2'))*10
				histDiff = (self.iterCgp.cell2ToCell1Feature(hist,mode='chi2'))*10
				print 'featMax',featDiff.min(),featDiff.max()
				print 'edgeMax',edge.min(),edge.max()
				print 'sift',siftDiff.min(),siftDiff.max()
				print 'hist',histDiff.min(),histDiff.max()
				edge+=0.1*featDiff
				edge+=1.0*siftDiff
				edge+=3.0*histDiff





				cuts=False
				e1=numpy.exp(-gamma*edge)
				e0=1.0-e1
				for ci in range(self.iterCgp.numCells(1)):
					size = len(self.iterCgp.cells1[ci].points)
					#print size
					e0[ci]*=float(size)
					e1[ci]*=float(size)
				
				
				for ci in range(self.iterCgp.numCells(1)):
					bb = len(self.iterCgp.cells1[ci].boundedBy)
					if bb==0 :
						print "ZERO BOUNDS \n\n"
						#e0[ci]*=float(size)
						e1[ci]+=2.0

				for ci in range(self.iterCgp.numCells(2)):
					size = len(self.iterCgp.cells1[ci].points)
					if size<=200 :
						boundedBy=numpy.array(self.iterCgp.cells2[ci].boundedBy)-1
						e1[boundedBy]+=2.0


				w = e1-e0

				if True:
				
					cgc,gm 	= multicutFromCgp2(cgp=self.iterCgp,e0=e0,e1=e1,parameter=opengm.InfParam(planar=True,inferMinMarginals=True))
					deleteN = 1#2*int(float(self.iterCgp.numCells(1))**(0.5)+0.5)
					#cgc.infer(cgc.verboseVisitor())
					cgc.infer()
					argDual = cgc.argDual()
					if(argDual.min()==1):
						print "READ GAMMA"
						gamma*=0.9
						continue
					else:
						cuts=True

					#cgp2d.visualize(self.rgbImage,cgp=self.iterCgp,edge_data_in=argDual.astype(numpy.float32))
					factorMinMarginals = cgc.factorMinMarginals()

					m0 = factorMinMarginals[:,0].astype(numpy.float128)
					m1 = factorMinMarginals[:,1].astype(numpy.float128)
					m0*=-1.0
					m1*=-1.0
					p0 =  numpy.exp(m0)/(numpy.exp(m0)+numpy.exp(m1))
					p1 =  numpy.exp(m1)/(numpy.exp(m0)+numpy.exp(m1))
					#cgp2d.visualize(self.rgbImage,cgp=self.iterCgp,edge_data_in=p1.astype(numpy.float32))

					whereOn = numpy.where(argDual==1)
					nOn   = len(whereOn[0])
					nOff  = len(p0)-nOn
					print "nOn",nOn,"off",nOff


					p1[whereOn]+=100.0
					sortedIndex = numpy.argsort(p1)

					toDelete = 1
					if deleteN > nOff:
						toDelete = nOff
					
					cellStates = numpy.ones(self.iterCgp.numCells(1),dtype=numpy.uint32)
					cellStates[sortedIndex[:toDelete]]=0
					#cellStates[numpy.argmax(w)]=0
					print "argmax"
				else :
					cellStates = numpy.ones(self.iterCgp.numCells(1),dtype=numpy.uint32)
					#cellStates[sortedIndex[:toDelete]]=0
					cellStates[numpy.argmax(w)]=0

				if self.iterCgp.numCells(2)<50:
					cgp2d.visualize(self.rgbImage,cgp=self.iterCgp)

				print "merge cells",self.iterCgp.numCells(2),self.iterCgp.numCells(1)
				newtgrid = self.iterCgp.merge2Cells(cellStates)
				self.iterCgp  = cgp2d.Cgp(newtgrid)










class CeMc(object):
	def __init__(self,cgp):
		self.cgp=cgp