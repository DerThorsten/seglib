#ifndef VIGRA_NODE_MAP_BASE
#define VIGRA_NODE_MAP_BASE

#include "graph_map_base.hxx"

namespace cgp2d{

template<class DYNAMIC_GRAPH,class CHILD_MAP,class T>
class DifferentialNodeMapBase{
public:
    DifferentialNodeMapBase(DynamicGraphType &);
};  

// implementation
template<class DYNAMIC_GRAPH>
inline DifferentialNodeMapBase<DYNAMIC_GRAPH>::DifferentialNodeMapBase(
    typename DifferentialNodeMapBase<DYNAMIC_GRAPH>::DynamicGraphType & graph
)
:   GraphMapBase<DYNAMIC_GRAPH>(graph){

}   







#endif /*VIGRA_NODE_MAP_BASE*/