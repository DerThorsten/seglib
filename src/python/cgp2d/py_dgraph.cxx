#define PY_ARRAY_UNIQUE_SYMBOL superimg_PyArray_API
#define NO_IMPORT_ARRAY


#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>

#include "seglib/cgp2d/cgp2d.hxx"
#include "seglib/cgp2d/dynamic_graph.hxx"
#include "seglib/cgp2d/cgp2d_python.hxx"
#include "seglib/cgp2d/objectives/high_level.hxx"
#include "seglib/distances/distance.hxx"


namespace cgp2d {

namespace boostp = boost::python;





void setInitalEdges(
    DynamicGraph & dgraph, 
    vigra::NumpyArray<2, LabelType > edges
){
    const size_t numberOfEdges = edges.shape(0);
    CGP_ASSERT_OP(numberOfEdges,==,dgraph.initNumberOfEdges());

    for(size_t edgeIndex=0;edgeIndex<numberOfEdges;++edgeIndex){
        dgraph.setInitalEdge(edgeIndex,edges(edgeIndex,0),edges(edgeIndex,1));
    }
}


struct EdgeMapBaseWrap : EdgeMapBase, boostp::wrapper<EdgeMapBase>
{
    void merge(const std::vector<size_t> & toMerge,const size_t newIndex){
        this->get_override("merge")(toMerge,newIndex);
    }
    void erase(const size_t index){
        this->get_override("erase")(index);
    }
};

struct NodeMapBaseWrap : NodeMapBase, boostp::wrapper<NodeMapBase>
{
    void merge(const std::vector<size_t> & toMerge,const size_t newIndex){
        this->get_override("merge")(toMerge,newIndex);
    }
};
/*
template<class VEC_MAP>
VEC_MAP * vecMapFactory(DynamicGraph & dgraph,const size_t size,const bool nodeMap){
    VEC_MAP * map = new VEC_MAP(dgraph,size,nodeMap);
    return map;
}
*/


template<class MAP_TYPE>
MAP_TYPE * featureMapConstructor(
    DynamicGraph & dgraph,
    vigra::MultiArrayView<2,typename  MAP_TYPE::value_type >    features,
    vigra::MultiArrayView<1,size_t>                             edgeSize
){
    return new MAP_TYPE(dgraph,features,edgeSize);
}


void export_dgraph()
{
    //using namespace python;
    
    boostp::docstring_options doc_options(true, true, false);


    ////////////////////////////////////////
    // Nodes Graph
    ////////////////////////////////////////
    // basic types
    // tgrid and input image type
    typedef Cgp<CoordinateType,LabelType> CgpType;
    typedef CgpType::TopologicalGridType TopologicalGridType;





    boostp::class_<DynamicGraph>("DynamicGraph",boostp::init<const size_t,const size_t >
        (
            ( 
                boostp::arg("numberOfNodes"),
                boostp::arg("numberOfEdges")
            )
        ) 
    )
        .def("initNumberOfNodes",&DynamicGraph::initNumberOfNodes,"get the initial number of nodes")
        .def("initNumberOfEdges",&DynamicGraph::initNumberOfEdges,"get the initial number of edges")
        .def("numberOfNodes",&DynamicGraph::numberOfNodes,"get the current number of nodes")
        .def("numberOfEdges",&DynamicGraph::numberOfEdges,"get the current number of edges")
        .def("getAndEdge",&DynamicGraph::getAndEdge,"get the edge with the smallest index")
        .def("setInitalEdge",&DynamicGraph::setInitalEdge,
            (
                boostp::arg("initEdge"),
                boostp::arg("initNode0"),
                boostp::arg("initNode1")
            ),
            "set inital edges (bevore andy merging)"
        )
        .def("setInitalEdges",vigra::registerConverters(&setInitalEdges),
            (
                boostp::arg("edges")
            ),
            "a  initNumberOfEdges x 2 array"
        )
        .def("mergeParallelEdges",&DynamicGraph::mergeParallelEdges,"merge parallel / double edges")
        .def("mergeRegions",&DynamicGraph::mergeRegions,"mergeTwoRegions")
    ;



    // map bases
    boostp::class_<EdgeMapBaseWrap, boost::noncopyable>("EdgeMapBase")
        .def("merge",boostp::pure_virtual(&EdgeMapBaseWrap::merge))
        .def("erase",boostp::pure_virtual(&EdgeMapBaseWrap::erase))
    ;

    boostp::class_<NodeMapBaseWrap, boost::noncopyable>("NodeMapBase")
        .def("merge",boostp::pure_virtual(&NodeMapBaseWrap::merge))
    ;



    typedef EdgeFeatureMap<float> EdgeFeatureMapFloat;


    
    boostp::class_<EdgeFeatureMapFloat, boostp::bases<EdgeMapBaseWrap> >(
        "EdgeFeatureMapFloat",boostp::init< >()
    )
        .def( "__init__",boostp::make_constructor(vigra::registerConverters(& featureMapConstructor<EdgeFeatureMapFloat>)))
    ;

    // factory
    //boostp::def("floatVecMap",&vecMapFactory<FloatVecMap>, boostp::return_value_policy<boostp::manage_new_object>() );
    
}

} // namespace vigra

