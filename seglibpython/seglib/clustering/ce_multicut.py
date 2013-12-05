from seglib import cgp2d 
from seglib.preprocessing import norm01
import opengm
import numpy
import vigra



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

	e0  = e
	e1  = 1.0-e0

	"""
	print "g ",gradient[:5]
	print "e0",e0[:5]
	print "e1",e1[:5]
	print "w ",(e0-e1)[:5]
	"""
	return e0-e1




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







def multicutFromCgp(cgp,weights=None):
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
	cgc = opengm.inference.Cgc(gm=gm,parameter=opengm.InfParam(planar=True))



	return cgc,gm



class CeMc(object):
	def __init__(self,cgp):
		self.cgp=cgp