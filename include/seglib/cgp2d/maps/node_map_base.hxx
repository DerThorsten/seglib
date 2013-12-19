#ifndef VIGRA_NODE_MAP_BASE
#define VIGRA_NODE_MAP_BASE

#include "graph_map_base.hxx"

namespace cgp2d{

template<class DYNAMIC_GRAPH>
class NodeMapBase{
public:
    NodeMapBase(DynamicGraphType &);
};  

// implementation
template<class DYNAMIC_GRAPH>
inline NodeMapBase<DYNAMIC_GRAPH>::NodeMapBase(
    typename NodeMapBase<DYNAMIC_GRAPH>::DynamicGraphType & graph
)
:   GraphMapBase<DYNAMIC_GRAPH>(graph){

}   







#endif /*VIGRA_NODE_MAP_BASE*/