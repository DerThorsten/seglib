    
#ifndef CGP2D_NODE_MAPS
#define CGP2D_NODE_MAPS

/* vigra */
#include <vigra/multi_array.hxx>
#include <vigra/tinyvector.hxx>

#include "seglib/cgp2d/cgp2d.hxx"
#include "seglib/cgp2d/cgp2d_python.hxx"
#include "seglib/cgp2d/dynamic_graph.hxx"
#include "seglib/distances/distance.hxx"
namespace cgp2d{

template<class T>
class NodeDiffMapBase  {
public:
	NodeDiffMapBase(){} //: NodeMapBase(){}
	//virtual void merge(const std::vector<size_t> & toMerge,const size_t newIndex)=0;
	virtual T nodeDistance(const size_t n0,const size_t n1)const=0;
	//virtual size_t nodeSize(const size_t n)const=0;
};



template<class T,class DIST_FUNCTOR>
class NodeFeatureMap : public NodeDiffMapBase<T> {

public:
	typedef T value_type;
	// dummy constructor
	NodeFeatureMap( DynamicGraph & dgrap):NodeDiffMapBase<T>(),dgraph_(dgrap){}
	// real constructor
	NodeFeatureMap( DynamicGraph & dgraph,vigra::MultiArrayView<2,T>  features, vigra::MultiArrayView<1,vigra::UInt32>  sizes,const float wardness);

	// virtual interface
	virtual void merge(const std::vector<size_t> & toMerge,const size_t newIndex);
	virtual T nodeDistance(const size_t n0,const size_t n1)const;
	virtual size_t nodeSize(const size_t n)const{return sizes_(n);}

	// additonal interface
	size_t numberOfFeatures()const{return featureBuffer_.shape(0);}
	void mapFeaturesToInitalNodes(vigra::MultiArrayView<2,T> & )const;
	const DynamicGraph & dgraph()const{return dgraph_;}
private:

	DynamicGraph & dgraph_;
	vigra::MultiArrayView<2,T>             features_;
	vigra::MultiArrayView<1,vigra::UInt32> sizes_;
	vigra::MultiArray<1,T> 				   featureBuffer_;	
	float wardness_;
};

template<class T,class DIST_FUNCTOR>
NodeFeatureMap<T,DIST_FUNCTOR>::NodeFeatureMap( DynamicGraph & dgraph, vigra::MultiArrayView<2,T>  features, vigra::MultiArrayView<1,vigra::UInt32>  sizes,const float wardness)
:	NodeDiffMapBase<T>(),
	dgraph_(dgraph),
	features_(features),
	sizes_(sizes),
	featureBuffer_(typename vigra::MultiArray<1,T>::difference_type(features.shape(1)) ),
	wardness_(wardness)
{
	CGP_ASSERT_OP(features_.shape(0),==,dgraph_.initNumberOfNodes());
	this->registerMap(dgraph_);
}
template<class T,class DIST_FUNCTOR>
void NodeFeatureMap<T,DIST_FUNCTOR>::merge(const std::vector<size_t> & toMerge,const size_t newIndex){

	featureBuffer_=static_cast<T>(0);
	vigra::UInt32 sizeSum=0;

	for(size_t i=0;i<toMerge.size();++i){
		const size_t tm=toMerge[i];
		const vigra::UInt32 s = sizes_(tm);
		sizeSum+=s;
		for(size_t f=0;f<numberOfFeatures();++f){
			featureBuffer_(f)+=static_cast<T>(s)*features_(tm,f);
		}
	}
	sizes_(newIndex)=sizeSum;
	for(size_t f=0;f<numberOfFeatures();++f){
		features_(newIndex,f)=featureBuffer_(f)/static_cast<T>(sizeSum);
	}
}

template<class T,class DIST_FUNCTOR>
T NodeFeatureMap<T,DIST_FUNCTOR>::nodeDistance(const size_t n0,const size_t n1)const{
	//dist::Norm<T> distFunctor;

	DIST_FUNCTOR distFunctor;
	const T dist    = distFunctor(features_.bindInner(n0),features_.bindInner(n1));
	const T rSizeA  = static_cast<T>(1.0f)/static_cast<T>(sizes_(n0));
	const T rSizeB  = static_cast<T>(1.0f)/static_cast<T>(sizes_(n1));
	const T wardDivisor = wardness_*(rSizeA+rSizeB) + (1.0-wardness_);
	return dist/(wardDivisor);
}

template<class T,class DIST_FUNCTOR>
void NodeFeatureMap<T,DIST_FUNCTOR>::mapFeaturesToInitalNodes(vigra::MultiArrayView<2,T> & result)const{
	for(size_t i=0;i<dgraph_.initNumberOfNodes();++i){
		const size_t rep=dgraph_.reprNode(i);
		for(size_t f=0;f<numberOfFeatures();++f){
			result(i,f)=features_(rep,f);
		}
	}
}

}

#endif  // CGP2D_NODE_MAPS
