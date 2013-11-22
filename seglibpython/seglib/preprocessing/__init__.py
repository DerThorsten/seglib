

def reshapeFromImage(data):
	dx,dy=data.shape[0:2]
	return data.reshape([dx*dy,-1])


def reshapeToImage(data,shape):
	dx,dy=shape[0:2]
	return data.reshape([dx,dy,-1])

def norm01(data):
	b=data.copy()
	b-=b.min()
	b/=b.max()
	return b

	

def normC01(data):
	assert data.ndim==3
	b=data.copy()
	for c in range(3):
		b[:,:,c]-=b[:,:,c].min()
		b[:,:,c]/=b[:,:,c].max()
	return b