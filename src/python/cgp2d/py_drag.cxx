#define PY_ARRAY_UNIQUE_SYMBOL superimg_PyArray_API
#define NO_IMPORT_ARRAY


#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>

#include "seglib/cgp2d/cgp2d.hxx"
#include "seglib/cgp2d/drag2d.hxx"
#include "seglib/cgp2d/cgp2d_python.hxx"
#include "seglib/cgp2d/objectives/high_level.hxx"
#include "seglib/distances/distance.hxx"


namespace cgp2d {

namespace python = boost::python;




template<class HLO>
vigra::NumpyAnyArray writeBackMergedFeatures(
    const HLO &  hlo , 
    vigra::NumpyArray<2, float > res = vigra::NumpyArray<2,float >()
){
    res.reshapeIfEmpty(hlo.features().shape());
    hlo.writeBackMergedFeatures(res);
    return res;
}




void export_drag()
{
    using namespace python;
    
    docstring_options doc_options(true, true, false);


    ////////////////////////////////////////
    // Nodes Graph
    ////////////////////////////////////////
    // basic types
    // tgrid and input image type
    typedef Cgp<CoordinateType,LabelType> CgpType;
    typedef CgpType::TopologicalGridType TopologicalGridType;


    typedef DynamicRag<CgpType> DragType;
    typedef typename DragType::NodesMapType NodesMapType;
    typedef typename DragType::EdgeMapType   EdgeMapType;


    python::class_<DynamicNodes>("DynamicNodes",python::init< >())
    ;
    python::class_<DynamicEdge>("DynamicEdge",python::init< >())
    ;

    class_< NodesMapType  >("NodesMap")
        .def(map_indexing_suite< NodesMapType  >())
    ;
    class_< EdgeMapType  >("EdgeMap")
        .def(map_indexing_suite< EdgeMapType  >())
    ;




    python::class_<DragType>("DynamicRag",python::init<const CgpType & >()
            [with_custodian_and_ward<1 /*custodian == self*/, 2 /*ward == const CgpType& */>()] )
        .def("numberOfNodes",&DragType::numberOfNodes,"get the current number of nodes")
        .def("numberOfEdges",&DragType::numberOfEdges,"get the current number of edges")
        .def("removeEdge",   &DragType::removeEdge,"remove edge")
        .def("nodes",        &DragType::nodeMap, return_internal_reference<>(),"get node map")
        .def("edges",        &DragType::edgeMap, return_internal_reference<>(),"get edge map")
    ;

}

} // namespace vigra

