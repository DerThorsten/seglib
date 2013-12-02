#define PY_ARRAY_UNIQUE_SYMBOL superimg_PyArray_API
#define NO_IMPORT_ARRAY

#include <string>
#include <cmath>

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>


#include <boost/array.hpp>

#include <boost/accumulators/accumulators.hpp>

#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/extended_p_square_quantile.hpp>
#include <boost/accumulators/statistics/tail_quantile.hpp>

#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>

#include "seglib/cgp2d/cgp2d.hxx"
#include "seglib/cgp2d/cgp2d_python.hxx"

namespace python = boost::python;

namespace cgp2d {

    // tgrid and input image type
    typedef Cgp<CoordinateType,LabelType> CgpType;
    typedef CgpType::TopologicalGridType TopologicalGridType;


    vigra::NumpyAnyArray  hlCost(

        const Cgp<CoordinateType,LabelType>  &  cgp,
        vigra::NumpyArray<1,LabelType>          argPrimal,
        vigra::NumpyArray<2,float>              features,
        vigra::NumpyArray<2,float>              mergedFeaturesBuffer,
    ){ 

    }



    void export_merge(){

        python::def("_hlCost",vigra::registerConverters(&hlCost),
            (
                python::arg("labelings"),
                python::arg("nLabels"),
                python::arg("out")=python::object()
            )
        );

    }

}