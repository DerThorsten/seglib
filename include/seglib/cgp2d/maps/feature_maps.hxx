    
#ifndef CGP2D_FEATURE_MAP
#define CGP2D_FEATURE_MAP

#include "seglib/cgp2d/cgp2d.hxx"
#include "seglib/cgp2d/cgp2d_python.hxx"
#include "seglib/cgp2d/dynamic_graph.hxx"
namespace cgp2d{

template<class T>
struct NodeDiffMapBase : public NodeMapBase {

	NodeDiffMapBase() : NodeMapBase(){}
	virtual void merge(const std::vector<size_t> & toMerge,const size_t newIndex)=0;
	virtual T nodeDistance(const size_t n0,const size_t n1)const=0;

};



template<class T>
class NodeFeatures : public NodeDiffMapBase<T> {

public:
	virtual void merge(const std::vector<size_t> & toMerge,const size_t newIndex);
	virtual T nodeDistance(const size_t n0,const size_t n1)const;
private:

};

}

#endif  // CGP2D_FEATURE_MAP
