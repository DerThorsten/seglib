#include <vigra/multi_convolution.hxx>


template<class T,class N>
class DenseHistogram{

public:
	typedef T ValueType;
	typedef T value_type;
	typedef vigra::MultiArray<T,N> StorateType;
	typedef typename StorateType::difference_type StorageDifferenceType;
	typedef vigra::TinyVector<T,N> ValueVector;

	DenseHistogram(){
	}

	DenseHistogram(
		const ValueVector 				&	min,
		const vValueVector 				&	max,
		const StorageDifferenceType  	&	bins
	)	:
		hist_(bins),
		min_(min),
		max_(max),
		fbin_(bins){
		// nothing more to do since hist store is allocated in initalzier list
	}


	void increment(const ValueVector & values , const ValueType weight = 1.0 ){
		hist_(bindIndex(values))+=weight
	}


	StorageDifferenceType binIndex(const ValueVector & values )const{
		//implement me
		ValueVector fIndex = values;
		fIndex-=min_;
		fIndex/=max_;
		fIndex*=fbin_;
		return StorageDifferenceType(fIndex);
	}


	// smooth inplace
	void smooth(const ValueType sigma){
		// perform isotropic Gaussian smoothing at scale 'sigma'
    	vigra::gaussianSmoothMultiArray(vigra::srcMultiArrayRange(hist_), destMultiArray(hist_), sigma);
	}

private:

	vigra::MultiArray<T,N>  hist_;
	ValueVector min_;
	ValueVector max_;
	ValueVector fbin_;
};



/*
	RGB   		=	10 *  10 * 10  = 1000 
	RG RB GB	=	3 * 10 * 10 = 300
	R G B       =    
*/ 