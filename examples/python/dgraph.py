import vigra
import numpy
from seglib import cgp2d 
import sys
from seglib.clustering.ce_multicut import *
from seglib.preprocessing import norm01,normCProb,reshapeToImage,normCProbFlat
from seglib.preprocessing import norm01,normCProb,reshapeToImage
from seglib.histogram import jointHistogram,histogram

img = "img/37073.jpg"
img="img/42049.jpg"
#img="/home/tbeier/src/seglib/examples/python/img/lena.bmp"
#img="/home/tbeier/src/seglib/examples/python/img/t.jpg"
img="/home/tbeier/src/privatOpengm/experiments/100075.jpg"
#img="/home/tbeier/src/privatOpengm/experiments/datasets/bsd500/BSR/BSDS500/data/images/train/118035.jpg"
#img="img/zebra.jpg"
img = numpy.squeeze(vigra.readImage(img))#[0:75,0:75,:]
lab = vigra.colors.transform_RGB2Lab(img)
labels ,nseg= vigra.analysis.slicSuperpixels(lab,10.0,7)
labels = vigra.analysis.labelImage(labels).astype(numpy.uint64)
#grad =vigra.filters.gaussianGradientMagnitude(lab,1.5)
#labels,nseg =vigra.analysis.watersheds(grad)
labels=labels.astype(numpy.uint64)

cgp,tgrid = cgp2d.cgpFromLabels(labels)
imgBig = vigra.sampling.resize(lab,cgp.shape)
imgBigRGB = vigra.sampling.resize(img,cgp.shape)
grad = numpy.squeeze(vigra.filters.gaussianGradientMagnitude(imgBig,2.5))




#cgp2d.visualize(imgBigRGB,cgp=cgp)





def visualizeRegionFeatures(cgp,features,cellType=2,useTopologicalShape=False):
	imgs=[]
	nF=features.shape[1]

	for f in range(nF):
		feat = features[:,f]
		res  = cgp.featureToImage(cellType=cellType,features=feat,useTopologicalShape=useTopologicalShape)
		imgs.append(res)

	out=numpy.array(imgs).astype(numpy.float32)
	print out.shape
	out=numpy.swapaxes(out,0,2)
	out=numpy.swapaxes(out,0,1)
	cgp2d.visualize(vigra.sampling.resize(out,cgp.shape+(3,)),cgp)




def tinyAggl(cgp,grad,imgBig,imgBigRGB,labSmall,beta):

	print "get hist"
	#hist  = normCProb(reshapeToImage(histogram( labSmall,r=3,sigma=[3.0,7.0]),labSmall.shape)) 

	print "get dgraph"
	dgraph=cgp2d.DynamicGraph(numberOfNodes=cgp.numCells(2),numberOfEdges=cgp.numCells(1))
	initEdges = cgp.cell1BoundsArray().astype(numpy.uint64)-1
	dgraph.setInitalEdges(initEdges)


	#print "create node maps A"
	#nodeFeat = cgp.accumulateCellFeatures(cellType=2,image=hist,features='Mean')[0]['Mean'].astype(numpy.float32)
	#nodeSizeA = cgp.cellSizes(2)
	#nodeMapA  = cgp2d.nodeFeatureMap(dgraph,nodeFeat,nodeSizeA,0.0,"chiSquared")

	#print "create node maps B"
	#nodeFeat = cgp.accumulateCellFeatures(cellType=2,image=imgBigRGB,features='Mean')[0]['Mean'].astype(numpy.float32)
	#nodeSizeB = cgp.cellSizes(2)
	#nodeMapB  = cgp2d.nodeFeatureMap(dgraph,nodeFeat,nodeSizeB,0.0,"norm")






	print "create edge maps"
	edgeFeat = cgp.accumulateCellFeatures(cellType=1,image=grad,features='Mean')[0]['Mean'].reshape(-1,1).astype(numpy.float32)
	edgeSize = cgp.cellSizes(1)
	edgeMap  = cgp2d.EdgeFeatureMap(dgraph,edgeFeat,edgeSize,beta,1.0)
	#edgeMap.registerNodeMap(nodeMapA,3.0)
	#edgeMap.registerNodeMap(nodeMapB,0.5)


	print "mergeParallelEdges"
	dgraph.mergeParallelEdges()

	c=1
	while(dgraph.numberOfEdges()>0):

		if c%100 == 0:
			print "reg / edges",dgraph.numberOfNodes(),dgraph.numberOfEdges()

		#activeEdgeLabels=dgraph.activeEdgeLabels()
		#argmin = numpy.argmin(feat[activeEdgeLabels])
		minEdge=edgeMap.minEdge()


		#anEdge=dgraph.getAndEdge()
		dgraph.mergeRegions(long(minEdge))
		c+=1

		if dgraph.numberOfNodes()==10 :
			state 				= dgraph.stateOfInitalEdges().astype(numpy.float32)
			cgp2d.visualize(imgBigRGB,cgp=cgp,edge_data_in=state,black=True,cmap='jet')
		if dgraph.numberOfNodes()==1 :
			state 				= dgraph.stateOfInitalEdges().astype(numpy.float32)
			ucmFeatures = edgeMap.computeUcmFeatures()
			cgp2d.visualize(imgBigRGB,cgp=cgp,edge_data_in=ucmFeatures,black=True,cmap='jet')
			return ucmFeatures


tinyAggl(cgp=cgp,grad=grad,imgBig=imgBig,imgBigRGB=imgBigRGB,labSmall=lab,beta=0.9)