    
#ifndef CGP2D_EDGE_MAPS_NEW
#define CGP2D_EDGE_MAPS_NEW

/* vigra */
#include <vigra/multi_array.hxx>
#include <vigra/tinyvector.hxx>

#include "seglib/cgp2d/cgp2d.hxx"
#include "seglib/cgp2d/cgp2d_python.hxx"
#include "seglib/cgp2d/dynamic_graph.hxx"
#include "seglib/distances/distance.hxx"
#include "seglib/cgp2d/maps/graph_map.hxx"
#include "seglib/cgp2d/dmin.hxx"
#include <functional>
namespace cgp2d{



template<class T>
class EdgeFeatureMap 
: public GraphMapBase<DynamicGraph> {
	typedef EdgeFeatureMap<T> SelfType;
public:
	typedef GraphMapBase<DynamicGraph> GraphMapBaseType;
	typedef DynamicGraph::MergeEdgeCallBack MergeEdgeCallBack;
	typedef DynamicGraph::EraseEdgeCallBack EraseEdgeCallBack;
	typedef DynamicGraph::NhChangedEdgeCallBack NhChangedEdgeCallBack;



	typedef DynamicGraph::EdgeType EdgeType;
	typedef DynamicGraph::NodeType NodeType;

	typedef T value_type;
	// dummy constructor
	EdgeFeatureMap( DynamicGraph & dgraph) : GraphMapBaseType(dgraph){}
	// real constructor
	EdgeFeatureMap( 
		DynamicGraph & dgraph,
		vigra::MultiArrayView<2,T>  features, 
		vigra::MultiArrayView<1,vigra::UInt32>  sizes,
		//NodeDiffMapBase<T> * nodeMap,
		const float beta,
		const float wardness
	);

	// callbacks
	void merge(const size_t a,const size_t b,const size_t r);
	void erase(const size_t index);
	void nhChange(const size_t index);

	T edgeWeight(const size_t e)const;

	// additonal interface
	size_t numberOfFeatures()const{return featureBuffer_.shape(0);}
	void computeUcmFeatures(vigra::MultiArrayView<1,T> & )const;
	size_t minEdge();



private:

	T  edgeFeatureSum(const size_t index)const{
		T sum = static_cast<T>(0);
		for(size_t f=0;f<numberOfFeatures();++f){
			sum+=features_(index,f);
		}
		return sum;
	}

	T  nodeDiffSum(const size_t n0,const size_t n1)const{
		return 0.0;
	}

	vigra::MultiArrayView<2,T>             features_;
	vigra::MultiArrayView<1,vigra::UInt32> sizes_;
	vigra::MultiArray<1,T> 				   featureBuffer_;	
	vigra::MultiArray<1,T> 				   weights_;	
	float beta_;
	float wardness_;
	minmax::FloatPq pq_; 
};

template<class T>
EdgeFeatureMap<T>::EdgeFeatureMap( 
	DynamicGraph & dgraph, 
	vigra::MultiArrayView<2,T>  features, 
	vigra::MultiArrayView<1,vigra::UInt32>  sizes,
	//NodeDiffMapBase<T> * nodeMap,
	const float beta,
	const float wardness
)
:	GraphMapBaseType(dgraph),
	features_(features),
	sizes_(sizes),
	featureBuffer_(typename vigra::MultiArray<1,T>::difference_type(features.shape(1)) ),
	weights_(typename vigra::MultiArray<1,T>::difference_type( dgraph.initNumberOfEdges() ) ),
	beta_(beta),
	wardness_(wardness),
	pq_( dgraph.initNumberOfEdges())
{
	// register callbacks
	{
  		MergeEdgeCallBack f ;
  		f = boost::bind(boost::mem_fn(&SelfType::merge), this , _1,_2,_3) ; //, _1, _2)
		this->graph().registerMergeEdgeCallBack(f);
	}
	{
  		EraseEdgeCallBack f ;
  		f = boost::bind(boost::mem_fn(&SelfType::erase), this , _1) ; //, _1, _2)
		this->graph().registgerEraseEdgeCallBack(f);
	}
	{
  		NhChangedEdgeCallBack f ;
  		f = boost::bind(boost::mem_fn(&SelfType::nhChange), this , _1) ; //, _1, _2)
		this->graph().registerNhChangedEdgeCallBack(f);
	}


	//this->registerMap(dgraph_);
	//nodeMaps_.push_back(nodeMap);
	// compute inital weights from edge features and region differences
	for(size_t edgeIndex=0;edgeIndex<this->graph().initNumberOfEdges();++edgeIndex){

		const EdgeType & edge 	= this->graph().getInitalEdge(edgeIndex);
		//const T nodeDist       	= nodeDiffSum(edge[0],edge[1]);
		const T pureEdgeWeight 	= edgeFeatureSum(edgeIndex);
		weights_(edgeIndex) = pureEdgeWeight;
		//weights_(edgeIndex) = beta_*pureEdgeWeight + (1.0-beta_)*nodeDist;
	}
	pq_.setValues(weights_.begin(),weights_.end());


}



template<class T>
void EdgeFeatureMap<T>::merge(const size_t a,const size_t b,const size_t r){
	if(a!=r && b!=r){
		CGP_ASSERT_OP(false,==,true);
	}
	featureBuffer_=static_cast<T>(0);
	vigra::UInt32 sizeSum=0;

	for(size_t i=0;i<2;++i){
		const size_t tm= (i==0? a : b);
		const vigra::UInt32 s = sizes_(tm);
		for(size_t f=0;f<numberOfFeatures();++f){
			featureBuffer_(f)+=static_cast<T>(s)*features_(tm,f);
		}
		sizeSum+=s;
		if(tm!=r){
			if(pq_.hasIndex(tm)){
				pq_.deleteIndex(tm);
			}
		}
	}
	sizes_(r)=sizeSum;
	for(size_t f=0;f<numberOfFeatures();++f){
		features_(r,f)=featureBuffer_(f)/static_cast<T>(sizeSum);
	}

	const T newEdgeWeight 	= edgeFeatureSum(r);
	pq_.changeValue(r,newEdgeWeight);
}

template<class T>
T EdgeFeatureMap<T>::edgeWeight(const size_t index)const{
	return weights_(index);
}

template<class T>
void EdgeFeatureMap<T>::erase(const size_t index){
	//std::cout<<"erase edge\n";
	if(pq_.hasIndex(index)){
		pq_.deleteIndex(index);
	}
}

template<class T>
void EdgeFeatureMap<T>::nhChange(const size_t index){
	//std::cout<<"nh change\n";
}

template<class T>
void EdgeFeatureMap<T>::computeUcmFeatures(vigra::MultiArrayView<1,T> & result)const{
	for(size_t i=0;i<this->graph().initNumberOfEdges();++i){
		const size_t rep=this->graph().reprEdge(i);
		result(i)=weights_(rep);
	}
}

template<class T>
size_t EdgeFeatureMap<T>::minEdge(){
	//std::cout<<"get min edge\n";
	size_t minIndex = pq_.minIndex();
	while(this->graph().hasEdge(minIndex)==false){
		pq_.deleteIndex(minIndex);
		minIndex=pq_.minIndex();
	}
	return minIndex;
}

}

#endif  // CGP2D_EDGE_MAPS_NEW
