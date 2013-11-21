import vigra
import seglib.cgp2d as cgp2d
import numpy
import opengm
from termcolor import colored
import sys
import gc

img 		= vigra.readImage('lena.bmp')
#img 		= vigra.sampling.resize(img,shape=[100,100])

scale 		= 1.5
gradMag 	= vigra.filters.gaussianGradientMagnitude(img,scale)
labels,nSeg = vigra.analysis.watersheds(gradMag)
labels		= numpy.require(labels,dtype=numpy.uint64)



tgrid 	= cgp2d.TopologicalGrid(labels)
cgp  	= cgp2d.Cgp(tgrid)


img 		= vigra.sampling.resize(img,shape=cgp.shape)






numVar 					= cgp.numCells(2)
numSecondOrderFactors 	= cgp.numCells(1)
numHighOrderFactors		= cgp.numCells(0)

print numVar,numSecondOrderFactors,numHighOrderFactors



cells0 = cgp.cells(0)
cells1 = cgp.cells(1)
cells2 = cgp.cells(2)


for cell0 in cells0:
	print cell0.label
	boundariesOfJunction = numpy.array(cell0.bounds)-1
	regionsOfJunction    = set()
	for boundaryIndex in boundariesOfJunction:

		boundaryCell = cells1[int(boundaryIndex)]
		r1,r2        = numpy.array(boundaryCell.bounds)-1

		#print r1,r2

		regionsOfJunction.add(r1)
		regionsOfJunction.add(r2)

	regionsOfJunction = sorted(list(regionsOfJunction))
	print regionsOfJunction