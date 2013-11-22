import vigra
import seglib.edge_detectors.pixel as edp

img 		= vigra.readImage('156065.jpg')#[0:75,0:75,:]

edp.textureGradient(img)

