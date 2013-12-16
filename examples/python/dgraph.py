import vigra
import numpy
from seglib import cgp2d 
import sys
from seglib.clustering.ce_multicut import *
from seglib.preprocessing import norm01,normCProb,reshapeToImage,normCProbFlat

img = "img/37073.jpg"
img="img/42049.jpg"
#img="/home/tbeier/src/seglib/examples/python/img/lena.bmp"
#img="/home/tbeier/src/seglib/examples/python/img/t.jpg"
#img="/home/tbeier/src/privatOpengm/experiments/100075.jpg"
#img="/home/tbeier/src/privatOpengm/experiments/datasets/bsd500/BSR/BSDS500/data/images/train/118035.jpg"
#img="img/zebra.jpg"
img = numpy.squeeze(vigra.readImage(img))#[0:75,0:75,:]
lab = vigra.colors.transform_RGB2Lab(img)
labels ,nseg= vigra.analysis.slicSuperpixels(lab,10.0,4)
labels = vigra.analysis.labelImage(labels).astype(numpy.uint64)
#grad =vigra.filters.gaussianGradientMagnitude(lab,1.5)
#labels,nseg =vigra.analysis.watersheds(grad)
labels=labels.astype(numpy.uint64)

cgp,tgrid = cgp2d.cgpFromLabels(labels)
imgBig = vigra.sampling.resize(lab,cgp.shape)
imgBigRGB = vigra.sampling.resize(img,cgp.shape)
grad = numpy.squeeze(vigra.filters.gaussianGradientMagnitude(imgBig,1.5))




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




def tinyAggl(cgp,grad,imgBig,imgBigRGB,beta):

	print "get dgraph"
	dgraph=cgp2d.DynamicGraph(numberOfNodes=cgp.numCells(2),numberOfEdges=cgp.numCells(1))
	initEdges = cgp.cell1BoundsArray().astype(numpy.uint64)-1
	dgraph.setInitalEdges(initEdges)

	print "create node maps A"
	nodeFeat = cgp.accumulateCellFeatures(cellType=2,image=imgBig,features='Mean')[0]['Mean'].astype(numpy.float32)
	nodeSize = numpy.ones(cgp.numCells(2),dtype=numpy.uint32)
	nodeMapA  = cgp2d.nodeFeatureMap(dgraph,nodeFeat,nodeSize,"norm")

	print "create node maps B"
	nodeFeat = cgp.accumulateCellFeatures(cellType=2,image=imgBigRGB,features='Mean')[0]['Mean'].astype(numpy.float32)
	nodeSize = numpy.ones(cgp.numCells(2),dtype=numpy.uint32)
	nodeMapB  = cgp2d.nodeFeatureMap(dgraph,nodeFeat,nodeSize,"norm")






	print "create edge maps"
	edgeFeat = cgp.accumulateCellFeatures(cellType=1,image=grad,features='Mean')[0]['Mean'].reshape(-1,1).astype(numpy.float32)
	#edgeFeat = norm01(edgeFeat)*0.99 + 0.005
	edgeSize = numpy.ones(cgp.numCells(1),dtype=numpy.uint32)


	print "edgeFeat",edgeFeat.shape,edgeFeat.dtype
	edgeFeat=numpy.array(edgeFeat)
	edgeMap  = cgp2d.EdgeFeatureMap(dgraph,edgeFeat,edgeSize,beta)
	edgeMap.registerNodeMap(nodeMapA)
	edgeMap.registerNodeMap(nodeMapB)


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

		if dgraph.numberOfNodes()==100 :
			state 				= dgraph.stateOfInitalEdges().astype(numpy.float32)
			cgp2d.visualize(imgBigRGB,cgp=cgp,edge_data_in=state,black=True,cmap='jet')
		if dgraph.numberOfNodes()==1 :
			state 				= dgraph.stateOfInitalEdges().astype(numpy.float32)
			ucmFeatures = edgeMap.computeUcmFeatures()
			cgp2d.visualize(imgBigRGB,cgp=cgp,edge_data_in=ucmFeatures,black=True,cmap='jet')
			return ucmFeatures


tinyAggl(cgp=cgp,grad=grad,imgBig=imgBig,imgBigRGB=imgBigRGB,beta=0.95)