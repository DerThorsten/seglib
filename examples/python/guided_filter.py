import vigra
import numpy
import pylab

from seglib import cgp2d 
from seglib.preprocessing import norm01
import seglib.edge_detectors.pixel as edp
import seglib.region_descriptors.pixel as rdp
from seglib.preprocessing import norm01,normC01
from seglib.histogram import jointHistogram,histogram
from seglib.region_descriptors.pixel.sift import denseSift


# change me to your path
img = "img/108005.jpg"
sigma = 2.0
epsilon = 0.6**2
img       = numpy.squeeze(vigra.readImage(img))#[0:75,0:75,:]



I = norm01(img)

#I  = img.copy()#norm01(vigra.colors.transform_RGB2Lab(img)[:,:,0])
p  = I.copy()


# 1)
meanI  = vigra.filters.gaussianSmoothing(I,sigma=sigma)
meanP  = vigra.filters.gaussianSmoothing(p,sigma=sigma)
coorI  = vigra.filters.gaussianSmoothing(I*I,sigma=sigma)
coorIP = vigra.filters.gaussianSmoothing(I*p,sigma=sigma)

# 2) 
varI   = coorI -meanI*meanI
covIP  = coorIP - meanI*meanP

# 3)
print "mima vari ", varI.min() ,varI.max()
varIA = numpy.abs(varI)
a  = covIP/(varI+epsilon)

aa = numpy.sum(a*meanI,axis=2)

aa = numpy.array(numpy.concatenate([aa[:,:,None]]*3,axis=2))


b  = numpy.array(meanP) - aa*numpy.array(meanI)

b  = vigra.taggedView(b,'xyc')

meanA = vigra.filters.gaussianSmoothing(a,sigma=sigma)
meanB = vigra.filters.gaussianSmoothing(b,sigma=sigma)

q = meanA*I + meanB

q = normC01(q)

toshow=[I,q]
f = pylab.figure()
for n, img in enumerate(toshow):
    #f.add_subplot(2, 1, n)  # this line outputs images on top of each other
    f.add_subplot(1, len(toshow), n)  # this line outputs images side-by-side
    pylab.imshow(numpy.swapaxes(norm01(img),0,1))
pylab.show()




