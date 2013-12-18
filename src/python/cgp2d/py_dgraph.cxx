#define PY_ARRAY_UNIQUE_SYMBOL superimg_PyArray_API
#define NO_IMPORT_ARRAY


#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>

#include "seglib/cgp2d/cgp2d.hxx"
#include "seglib/cgp2d/cgp2d_python.hxx"

#include "seglib/cgp2d/dynamic_graph.hxx"
#include "seglib/cgp2d/maps/node_maps.hxx"
#include "seglib/cgp2d/maps/edge_maps.hxx"
#include "seglib/distances/distance.hxx"

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

/*
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

template<class T>
struct NodeDiffMapBaseWrap : NodeDiffMapBase<T>, boostp::wrapper<NodeDiffMapBase<T> > 
{

    void merge(const std::vector<size_t> & toMerge,const size_t newIndex){
        this->get_override("merge")(toMerge,newIndex);
    }
    T  nodeDistance(const size_t n0,const size_t n1)const{
        this->get_override("nodeDistance")(n0,n1);
    }
    size_t nodeSize(const size_t n)const{
        this->get_override("nodeSize")(n);
    }
};

template<class T>
struct EdgeWeightBaseWrap : EdgeWeightMapBase<T>, boostp::wrapper<EdgeWeightMapBase<T> >
{
    void merge(const std::vector<size_t> & toMerge,const size_t newIndex){
        this->get_override("merge")(toMerge,newIndex);
    }
    void erase(const size_t index,const size_t newNodeIndex){
        this->get_override("erase")(index,newNodeIndex);
    }
    T  edgeWeight(const size_t e)const{
        this->get_override("edgeWeight")(e);
    }
};
*/


/*
template<class VEC_MAP>
VEC_MAP * vecMapFactory(DynamicGraph & dgraph,const size_t size,const bool nodeMap){
    VEC_MAP * map = new VEC_MAP(dgraph,size,nodeMap);
    return map;
}
*/


template<class MAP_TYPE,unsigned int FDIM>
NodeDiffMapBase<typename  MAP_TYPE::value_type>  * nodeFeatureMapConstructor(
    DynamicGraph & dgraph,
    vigra::MultiArrayView<FDIM,typename  MAP_TYPE::value_type >  features,
    vigra::MultiArrayView<1,vigra::UInt32>                       edgeSize,
    const float wardness
){
    return new MAP_TYPE(dgraph,features,edgeSize,wardness);
}


template<class MAP_TYPE,unsigned int FDIM>
MAP_TYPE * edgeFeatureMapConstructor(
    DynamicGraph & dgraph,
    vigra::MultiArrayView<FDIM,typename  MAP_TYPE::value_type >    features,
    vigra::MultiArrayView<1,vigra::UInt32>                         edgeSize,
    //NodeDiffMapBase<typename  MAP_TYPE::value_type>            *   nodeMap,
    const float beta,
    const float wardness
){
    return new MAP_TYPE(dgraph,features,edgeSize,beta,wardness);
}


template<class MAP_TYPE>
vigra::NumpyAnyArray computeUcmFeatures(
    const MAP_TYPE & map,
    vigra::NumpyArray<1 , float> res = vigra::NumpyArray<1,float >()
){
    const size_t numberOfEdges=map.graph().initNumberOfEdges();
    res.reshapeIfEmpty(vigra::NumpyArray<1 , float>::difference_type(numberOfEdges));
    map.computeUcmFeatures(res);
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



template<class MAP,class BASE_MAP>
void export_node_feature_map(const std::string & clsName){

    boostp::class_<MAP, boostp::bases< BASE_MAP >  >(
        clsName.c_str(),boostp::init< DynamicGraph & >()
    )
        .def( "__init__",boostp::make_constructor(vigra::registerConverters(& nodeFeatureMapConstructor<MAP,2>)))
        .def("mapFeaturesToInitalNodes",vigra::registerConverters(&mapFeaturesToInitalNodes<MAP>),
            (
                boostp::arg("out")=boostp::object()
            ),
            "map merged node features to inital nodes"
        )
    ;
}

template<class T>
NodeDiffMapBase<T>  * nodeFeatureMapFactory(
    DynamicGraph & dgraph,
    vigra::MultiArrayView<2, T >  features,
    vigra::MultiArrayView<1,vigra::UInt32>                       edgeSize,
    const float wardness,
    const std::string & distance
){
    if(distance==std::string("norm")){
        typedef dist::Norm<T> DistFunctor;
        typedef NodeFeatureMap<T,DistFunctor > NodeMap;
        return new NodeMap(dgraph,features,edgeSize,wardness);
    }
    else if(distance==std::string("squaredNorm")){
        typedef dist::SquaredNorm<T> DistFunctor;
        typedef NodeFeatureMap<T,DistFunctor > NodeMap;
        return new NodeMap(dgraph,features,edgeSize,wardness);
    }
    else if(distance==std::string("chiSquared")){
        typedef dist::ChiSquared<T> DistFunctor;
        typedef NodeFeatureMap<T,DistFunctor > NodeMap;
        return new NodeMap(dgraph,features,edgeSize,wardness);
    }
    else{
        throw std::runtime_error("unknown distance");
    }    
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





    boostp::class_<DynamicGraph,boost::noncopyable>("DynamicGraph",boostp::init<const size_t,const size_t >
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

    /*

    // map bases
    boostp::class_<EdgeMapBaseWrap, boost::noncopyable>("EdgeMapBase")
        .def("merge",boostp::pure_virtual(&EdgeMapBaseWrap::merge))
        .def("erase",boostp::pure_virtual(&EdgeMapBaseWrap::erase))
    ;

    boostp::class_<NodeMapBaseWrap, boost::noncopyable >("NodeMapBase")
        .def("merge",boostp::pure_virtual(&NodeMapBaseWrap::merge))
    ;

    typedef NodeDiffMapBaseWrap<float> NodeDiffMapBaseWrapFloat;
    boostp::class_<NodeDiffMapBaseWrapFloat, boost::noncopyable >("NodeDiffMapBase")
        .def("merge",boostp::pure_virtual(&NodeDiffMapBaseWrapFloat::merge))
        .def("nodeDistance",boostp::pure_virtual(&NodeDiffMapBaseWrapFloat::nodeDistance))
        .def("nodeSize",boostp::pure_virtual(&NodeDiffMapBaseWrapFloat::nodeSize))
    ;

    typedef EdgeWeightBaseWrap<float> EdgeWeightBaseWrapFloat;
    boostp::class_<EdgeWeightBaseWrapFloat, boost::noncopyable >("EdgeWeightBase")
        .def("merge",boostp::pure_virtual(&EdgeWeightBaseWrapFloat::merge))
        .def("erase",boostp::pure_virtual(&EdgeWeightBaseWrapFloat::erase))
        .def("edgeWeight",boostp::pure_virtual(&EdgeWeightBaseWrapFloat::edgeWeight))
    ;
 

    // export factory
    boostp::def("nodeFeatureMap",vigra::registerConverters(&nodeFeatureMapFactory<float>), boostp::return_value_policy<boostp::manage_new_object>());


    typedef NodeFeatureMap<float,dist::Norm<float> >        NodeFeatureMapNormFloat;
    typedef NodeFeatureMap<float,dist::SquaredNorm<float> > NodeFeatureMapSquaredNormFloat;
    typedef NodeFeatureMap<float,dist::ChiSquared<float> >  NodeFeatureMapChiSquaredFloat;
    export_node_feature_map<NodeFeatureMapNormFloat,        NodeDiffMapBaseWrapFloat>("NodeFeatureMapNorm");
    export_node_feature_map<NodeFeatureMapSquaredNormFloat, NodeDiffMapBaseWrapFloat>("NodeFeatureMapSquaredNorm");
    export_node_feature_map<NodeFeatureMapChiSquaredFloat,  NodeDiffMapBaseWrapFloat>("NodeFeatureMapChiSquared");
    */


    typedef EdgeFeatureMap<float> EdgeFeatureMapFloat;
    boostp::class_<EdgeFeatureMapFloat  >(
        "EdgeFeatureMap",boostp::init< DynamicGraph & >()
    )
        .def( "__init__",boostp::make_constructor(vigra::registerConverters(& edgeFeatureMapConstructor<EdgeFeatureMapFloat,2>)))
        .def("minEdge",&EdgeFeatureMapFloat::minEdge,"get the edge with minimum edge indicator")
        .def("computeUcmFeatures",vigra::registerConverters(&computeUcmFeatures<EdgeFeatureMapFloat>),
            (
                boostp::arg("out")=boostp::object()
            ),
            "compute ucm transformation"
        )
        //.def("registerNodeMap",&EdgeFeatureMapFloat::registerNodeMap)
    ;
    
    //boostp::def("floatVecMap",&vecMapFactory<FloatVecMap>, boostp::return_value_policy<boostp::manage_new_object>() );
    
}

} // namespace vigra

