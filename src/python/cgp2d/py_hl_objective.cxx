#define PY_ARRAY_UNIQUE_SYMBOL superimg_PyArray_API
#define NO_IMPORT_ARRAY


#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>

#include "seglib/cgp2d/cgp2d.hxx"
#include "seglib/cgp2d/cgp2d_python.hxx"
#include "seglib/cgp2d/objectives/high_level.hxx"
#include "seglib/distances/distance.hxx"


namespace cgp2d {

namespace python = boost::python;



template<class HLO>
void setRegionFeatures(
    HLO &  hlo , 
    vigra::NumpyArray<2,float> cell2Features
){
    hlo.setRegionFeatures(cell2Features);
}


template<class HLO>
void setPrimalLabels(
    HLO &  hlo , 
    vigra::NumpyArray<1,LabelType> labels
){
    hlo.setPrimalLabels(labels);
}



template<class HLO>
float betweenClusterDistance(
    HLO &  hlo , 
    const std::string distance,
    const double gamma
){
    if(distance=="squaredNorm"){
        dist::ChiSquared<double> distFunctor;
        return hlo.betweenClusterDistance(distFunctor,gamma);
    }
    else if(distance=="chi2"){
        dist::SquaredNorm<double> distFunctor;
        return hlo.betweenClusterDistance(distFunctor,gamma);
    }
}

template<class HLO>
vigra::NumpyAnyArray writeBackMergedFeatures(
    const HLO &  hlo , 
    vigra::NumpyArray<2, float > res = vigra::NumpyArray<2,float >()
){
    res.reshapeIfEmpty(hlo.features().shape());
    hlo.writeBackMergedFeatures(res);
    return res;
}




void export_hl_objective()
{
    using namespace python;
    
    docstring_options doc_options(true, true, false);


    ////////////////////////////////////////
    // Region Graph
    ////////////////////////////////////////
    // basic types
    // tgrid and input image type
    typedef Cgp<CoordinateType,LabelType> CgpType;
    typedef CgpType::TopologicalGridType TopologicalGridType;


    typedef HighLevelObjective<CgpType,float> HlOjective;


    python::class_<HlOjective>("HighLevelObjective",python::init<const CgpType & >()
            [with_custodian_and_ward<1 /*custodian == self*/, 2 /*ward == const CgpType& */>()] )
        .def("setRegionFeatures", vigra::registerConverters(&setRegionFeatures<HlOjective>)  )
        .def("setPrimalLabels",   vigra::registerConverters(&setPrimalLabels<HlOjective>)  )
        .def("mergeFeatures",&HlOjective::mergeFeatures)
        .def("withinClusterDistance",&HlOjective::withinClusterDistance)
        .def("betweenClusterDistance",&betweenClusterDistance<HlOjective>)
        .def("writeBackMergedFeatures",vigra::registerConverters(&writeBackMergedFeatures<HlOjective>),
            (
                arg("res")=python::object()
            )
        )
    ;

}

} // namespace vigra

