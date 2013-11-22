# own code 
from seglib.preprocessing 	import reshapeFromImage,reshapeToImage
from seglib.clustering 		import MiniBatchKMeans

#external
from sklearn.decomposition import PCA,ProbabilisticPCA, RandomizedPCA, KernelPCA, SparsePCA, TruncatedSVD ,NMF


def imageCodebookClustering(img,nClusters,distance='euclidean',batchSize=1000,iterations=500,nInit=10):
    shape 	= img.shape
    X 		= reshapeFromImage(img)
    kmeans 	= MiniBatchKMeans(nClusters=nClusters,batchSize=batchSize,iterations=iterations,nInit=nInit)
    kmeans.fit(X)


    _distToCenters 	= reshapeToImage(kmeans._distToCenters,shape)
    _labels 		= reshapeToImage(kmeans._labels,shape)
    _centers 		= kmeans._centers

    return _labels,_centers,_distToCenters



def imageDimensionReduction(img,nComponents=3,alg='pca',**kwargs):
    shape   = img.shape
    X       = reshapeFromImage(img)
    algs    = dict(pca=PCA,ppca=ProbabilisticPCA,rpca=RandomizedPCA,kpca=KernelPCA,spca=SparsePCA,tsvd=TruncatedSVD,nmf=NMF)
    salg    = algs[alg](n_components=nComponents,**kwargs)

    # reduce dimensionality 
    XX = reshapeToImage(salg.fit_transform(X),shape)
    return XX
