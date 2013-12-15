#define PY_ARRAY_UNIQUE_SYMBOL superimg_PyArray_API
#define NO_IMPORT_ARRAY


#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>

#include "seglib/cgp2d/cgp2d.hxx"
#include "seglib/cgp2d/cgp2d_python.hxx"

#include "seglib/cgp2d/dynamic_graph.hxx"
#include "seglib/cgp2d/maps/feature_map.hxx"


namespace cgp2d {

namespace boostp = boost::python;







vigra::NumpyAnyArray stateOfInitalEdges(
    const DynamicGraph & dgraph, 
    vigra::NumpyArray<1 , LabelType> res = vigra::NumpyArray<1,LabelType >()
){
    const size_t initNumberOfEdges=dgraph.initNumberOfEdges();
    res.reshapeIfEmpty(vigra::NumpyArray<1 , LabelType>::difference_type(initNumberOfEdges));
    dgraph.stateOfInitalEdges(res.begin(),res.end());
    return res;
}


vigra::NumpyAnyArray activeEdgeLabels(
    const DynamicGraph & dgraph, 
    vigra::NumpyArray<1 , LabelType> res = vigra::NumpyArray<1,LabelType >()
){
    const size_t numberOfEdges=dgraph.numberOfEdges();
    res.reshapeIfEmpty(vigra::NumpyArray<1 , LabelType>::difference_type(numberOfEdges));
    dgraph.activeEdgeLabels(res.begin(),res.end());
    return res;
}


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
    void erase(const size_t index,const size_t newNodeIndex){
        this->get_override("erase")(index,newNodeIndex);
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


template<class MAP_TYPE,unsigned int FDIM>
MAP_TYPE * featureMapConstructor(
    DynamicGraph & dgraph,
    vigra::MultiArrayView<FDIM,typename  MAP_TYPE::value_type >    features,
    vigra::MultiArrayView<1,size_t>                                 edgeSize
){
    return new MAP_TYPE(dgraph,features,edgeSize);
}


template<class MAP_TYPE,class OTHER_MAP_TYPE,unsigned int FDIM>
MAP_TYPE * featureMapConstructor2(
    DynamicGraph & dgraph,
    const OTHER_MAP_TYPE& otherMap,
    vigra::MultiArrayView<FDIM,typename  MAP_TYPE::value_type >    features,
    vigra::MultiArrayView<1,size_t>                                 edgeSize,
    const float beta
){
    return new MAP_TYPE(dgraph,otherMap,features,edgeSize,beta);
}


template<class MAP_TYPE>
vigra::NumpyAnyArray computeUcmFeatures(
    const MAP_TYPE & map,
    vigra::NumpyArray<1 , float> res = vigra::NumpyArray<1,float >()
){
    const size_t numberOfEdges=map.dgraph().initNumberOfEdges();
    res.reshapeIfEmpty(vigra::NumpyArray<1 , float>::difference_type(numberOfEdges));
    map.computeUcmFeatures(res.begin(),res.end());
    return res;
}

template<class MAP_TYPE>
vigra::NumpyAnyArray mapFeaturesToInitalNodes(
    const MAP_TYPE & map,
    vigra::NumpyArray<2 , float> res = vigra::NumpyArray<2,float >()
){
    const size_t numberOfNodes=map.dgraph().initNumberOfNodes();
    const size_t numberOfFeatures = map.numberOfFeatures();
    res.reshapeIfEmpty(vigra::NumpyArray<2 , float>::difference_type(numberOfNodes,numberOfFeatures));
    map.mapFeaturesToInitalNodes(res);
    return res;
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
        .def("stateOfInitalEdges",vigra::registerConverters(&stateOfInitalEdges),
            (
                boostp::arg("out")=boostp::object()
            ),
            "get edge state remapped on inital edges"
        )
        .def("activeEdgeLabels",vigra::registerConverters(&activeEdgeLabels),
            (
                boostp::arg("out")=boostp::object()
            ),
            "activeEdgeLabels"
        )
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
    typedef EdgeUcmMap<float>     EdgeUcmMap;
    typedef NodeFeatureMap<float> NodeFeatureMapFloat;
    typedef DiffEdgeMap<float>    DiffEdgeMap;

    boostp::class_<EdgeFeatureMapFloat, boostp::bases<EdgeMapBaseWrap> >(
        "EdgeFeatureMapFloat",boostp::init< >()
    )
        .def( "__init__",boostp::make_constructor(vigra::registerConverters(& featureMapConstructor<EdgeFeatureMapFloat,2>)))
    ;

    boostp::class_<EdgeUcmMap, boostp::bases<EdgeMapBaseWrap> >(
        "EdgeUcmMap",boostp::init< DynamicGraph & >()
    )
        .def( "__init__",boostp::make_constructor(vigra::registerConverters(& featureMapConstructor<EdgeUcmMap,1>)))
        .def("minEdge",&EdgeUcmMap::minEdge,"get the edge with minimum edge indicator")
        .def("computeUcmFeatures",vigra::registerConverters(&computeUcmFeatures<EdgeUcmMap>),
            (
                boostp::arg("out")=boostp::object()
            ),
            "compute ucm transformation"
        )
    ;

    boostp::class_<DiffEdgeMap, boostp::bases<EdgeMapBaseWrap> >(
        "DiffEdgeMap",boostp::init< DynamicGraph & ,NodeFeatureMapFloat >()
    )
        .def( "__init__",boostp::make_constructor(vigra::registerConverters(& featureMapConstructor2<DiffEdgeMap,NodeFeatureMapFloat,1>)))
        .def("minEdge",&DiffEdgeMap::minEdge,"get the edge with minimum edge indicator")
        .def("computeUcmFeatures",vigra::registerConverters(&computeUcmFeatures<DiffEdgeMap>),
            (
                boostp::arg("out")=boostp::object()
            ),
            "compute ucm transformation"
        )
    ;

    boostp::class_<NodeFeatureMapFloat, boostp::bases<NodeMapBaseWrap> >(
        "NodeFeatureMapFloat",boostp::init< DynamicGraph & >()
    )
        .def( "__init__",boostp::make_constructor(vigra::registerConverters(& featureMapConstructor<NodeFeatureMapFloat,2>)))
        .def("mapFeaturesToInitalNodes",vigra::registerConverters(&mapFeaturesToInitalNodes<NodeFeatureMapFloat>),
            (
                boostp::arg("out")=boostp::object()
            ),
            "map merged node features to inital nodes"
        )

    ;

    // factory
    //boostp::def("floatVecMap",&vecMapFactory<FloatVecMap>, boostp::return_value_policy<boostp::manage_new_object>() );
    
}

} // namespace vigra

