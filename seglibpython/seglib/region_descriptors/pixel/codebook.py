# own code 
from seglib.preprocessing import reshapeFromImage,reshapeToImage
from seglib.clustering 	  import MiniBatchKMeans as MiniBatchKMeansCustom
from sklearn.cluster      import MiniBatchKMeans as MiniBatchKMeansSklearn
#external
from sklearn.decomposition import PCA,ProbabilisticPCA, RandomizedPCA, KernelPCA, SparsePCA, TruncatedSVD ,NMF
from scipy.spatial.distance import cdist

def imageCodebookClustering(img,nClusters,distance=None,batchSize=100,iterations=100,nInit=10):
    """

    """
    shape 	= img.shape
    X 		= reshapeFromImage(img)

    if distance is None:
        kmeans  = MiniBatchKMeansSklearn(nClusters,max_iter=iterations,batch_size=batchSize)
        kmeans.fit(X)
        labels= reshapeToImage(kmeans.labels_,shape)
        centers = kmeans.cluster_centers_ 
        distToCenters = cdist(X,centers)
        distToCenters  = reshapeToImage(distToCenters,shape)
    else :
        kmeans 	= MiniBatchKMeansCustom(nClusters=nClusters,batchSize=batchSize,iterations=iterations,nInit=nInit)
        kmeans.fit(X)


        distToCenters 	= reshapeToImage(kmeans._distToCenters,shape)
        labels 		= reshapeToImage(kmeans._labels,shape)
        centers 		= kmeans._centers

    return labels,centers,distToCenters



def imageDimensionReduction(img,nComponents=3,alg='pca',**kwargs):
    shape   = img.shape
    X       = reshapeFromImage(img)
    algs    = dict(pca=PCA,ppca=ProbabilisticPCA,rpca=RandomizedPCA,kpca=KernelPCA,spca=SparsePCA,tsvd=TruncatedSVD,nmf=NMF)
    salg    = algs[alg](n_components=nComponents,**kwargs)

    # reduce dimensionality 
    XX = reshapeToImage(salg.fit_transform(X),shape)
    return XX
