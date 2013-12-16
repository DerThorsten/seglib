    
#ifndef CGP2D_EDGE_MAPS
#define CGP2D_EDGE_MAPS

/* vigra */
#include <vigra/multi_array.hxx>
#include <vigra/tinyvector.hxx>

#include "seglib/cgp2d/cgp2d.hxx"
#include "seglib/cgp2d/cgp2d_python.hxx"
#include "seglib/cgp2d/dynamic_graph.hxx"
#include "seglib/distances/distance.hxx"
#include "seglib/cgp2d/dmin.hxx"
namespace cgp2d{



template<class T>
class EdgeWeightMapBase : public EdgeMapBase {
public:
	EdgeWeightMapBase() : EdgeMapBase(){}
	virtual void merge(const std::vector<size_t> & toMerge,const size_t newIndex)=0;
	virtual void erase(const size_t index,const size_t newNodeIndex)=0;
	virtual T edgeWeight(const size_t e)const=0;
};




template<class T>
class EdgeFeatureMap : public EdgeWeightMapBase<T> {

public:
	typedef DynamicGraph::EdgeType EdgeType;
	typedef DynamicGraph::NodeType NodeType;

	typedef T value_type;
	// dummy constructor
	EdgeFeatureMap( DynamicGraph & dgrap):EdgeWeightMapBase<T>(),dgraph_(dgrap){}
	// real constructor
	EdgeFeatureMap( 
		DynamicGraph & dgraph,vigra::MultiArrayView<2,T>  features, 
		vigra::MultiArrayView<1,vigra::UInt32>  sizes,
		//NodeDiffMapBase<T> * nodeMap,
		const float beta
	);

	// virtual interface
	virtual void merge(const std::vector<size_t> & toMerge,const size_t newIndex);
	virtual void erase(const size_t index,const size_t newNodeIndex);
	virtual T edgeWeight(const size_t e)const;

	// additonal interface
	size_t numberOfFeatures()const{return featureBuffer_.shape(0);}
	void computeUcmFeatures(vigra::MultiArrayView<1,T> & )const;
	const DynamicGraph & dgraph()const{return dgraph_;}

	size_t minEdge();

	void registerNodeMap( NodeDiffMapBase<T> * nodeMap){
		nodeMaps_.push_back(nodeMap);
	}

private:

	T  edgeFeatureSum(const size_t index)const{
		T sum = static_cast<T>(0);
		for(size_t f=0;f<numberOfFeatures();++f){
			sum+=features_(index,f);
		}
		return sum;
	}

	T  nodeDiffSum(const size_t n0,const size_t n1)const{

		T sum = static_cast<T>(0);
		for(size_t m=0;m<nodeMaps_.size();++m){
			sum+=nodeMaps_[m]->nodeDistance(n0,n1);
		}
		return sum;
	}

	DynamicGraph & dgraph_;
	vigra::MultiArrayView<2,T>             features_;
	vigra::MultiArrayView<1,vigra::UInt32> sizes_;
	vigra::MultiArray<1,T> 				   featureBuffer_;	

	std::vector< NodeDiffMapBase<T> *  >   nodeMaps_;

	vigra::MultiArray<1,T> 				   weights_;	
	float beta_;
	minmax::FloatPq pq_; 
};

template<class T>
EdgeFeatureMap<T>::EdgeFeatureMap( 
	DynamicGraph & dgraph, 
	vigra::MultiArrayView<2,T>  features, 
	vigra::MultiArrayView<1,vigra::UInt32>  sizes,
	//NodeDiffMapBase<T> * nodeMap,
	const float beta
)
:	EdgeWeightMapBase<T>(),
	dgraph_(dgraph),
	features_(features),
	sizes_(sizes),
	featureBuffer_(typename vigra::MultiArray<1,T>::difference_type(features.shape(1)) ),
	weights_(typename vigra::MultiArray<1,T>::difference_type( dgraph_.initNumberOfEdges() ) ),
	beta_(beta),
	pq_( dgraph_.initNumberOfEdges())
{
	this->registerMap(dgraph_);
	//nodeMaps_.push_back(nodeMap);
	// compute inital weights from edge features and region differences
	for(size_t edgeIndex=0;edgeIndex<dgraph_.initNumberOfEdges();++edgeIndex){

		const EdgeType & edge 	= dgraph_.getInitalEdge(edgeIndex);
		const T nodeDist       	= nodeDiffSum(edge[0],edge[1]);
		const T pureEdgeWeight 	= edgeFeatureSum(edgeIndex);
		weights_(edgeIndex) = beta_*pureEdgeWeight + (1.0-beta_)*nodeDist;
	}
	pq_.setValues(weights_.begin(),weights_.end());
}



template<class T>
void EdgeFeatureMap<T>::merge(const std::vector<size_t> & toMerge,const size_t newIndex){
	//std::cout<<"merge\n";
	featureBuffer_=static_cast<T>(0);
	vigra::UInt32 sizeSum=0;

	for(size_t i=0;i<toMerge.size();++i){
		const size_t tm=toMerge[i];
		const vigra::UInt32 s = sizes_(tm);
		for(size_t f=0;f<numberOfFeatures();++f){
			featureBuffer_(f)+=static_cast<T>(s)*features_(tm,f);
		}
		sizeSum+=s;
		if(tm!=newIndex){
			if(pq_.hasIndex(tm)){
				pq_.deleteIndex(tm);
			}
		}
	}
	sizes_(newIndex)=sizeSum;
	for(size_t f=0;f<numberOfFeatures();++f){
		features_(newIndex,f)=featureBuffer_(f)/static_cast<T>(sizeSum);
	}
}

template<class T>
T EdgeFeatureMap<T>::edgeWeight(const size_t index)const{
	return weights_(index);
}

template<class T>
void EdgeFeatureMap<T>::erase(const size_t index,const size_t newNodeIndex){
	//std::cout<<"erase\n";
	if(pq_.hasIndex(index)){
		pq_.deleteIndex(index);
	}
	const NodeType & node = dgraph_.getNode(newNodeIndex);
	for(typename NodeType::EdgeIterator iter=node.edgesBegin();iter!=node.edgesEnd();++iter){
		const size_t edgeIndex = *iter;
		const EdgeType & edge = dgraph_.getEdge(edgeIndex);
		const T nodeDist      = nodeDiffSum(edge[0],edge[1]);

		//std::cout<<"node dist "<<nodeDist<<"\n";

		const T pureEdgeWeight 	= edgeFeatureSum(edgeIndex);
		weights_(edgeIndex) = beta_*pureEdgeWeight + (1.0-beta_)*nodeDist;
		pq_.changeValue(edgeIndex,weights_(edgeIndex));
	}
}

template<class T>
void EdgeFeatureMap<T>::computeUcmFeatures(vigra::MultiArrayView<1,T> & result)const{
	for(size_t i=0;i<dgraph_.initNumberOfEdges();++i){
		const size_t rep=dgraph_.reprEdge(i);
		result(i)=weights_(rep);
	}
}

template<class T>
size_t EdgeFeatureMap<T>::minEdge(){
	//std::cout<<"get min edge\n";
	size_t minIndex = pq_.minIndex();
	while(dgraph_.hasEdge(minIndex)==false){
		pq_.deleteIndex(minIndex);
		minIndex=pq_.minIndex();
	}
	return minIndex;
}

}

#endif  // CGP2D_EDGE_MAPS
