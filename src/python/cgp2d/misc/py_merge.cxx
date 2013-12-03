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


    void mergeFeatures(
        vigra::NumpyArray<1,LabelType>          labeling,
        const size_t                            numberOfLabels,
        vigra::NumpyArray<1,float>              weights,
        vigra::NumpyArray<2,float>              features,
        vigra::NumpyArray<2,float>              mergedFeatures,
        vigra::NumpyArray<1,float>              weightBuffer
    ){ 
        CGP_ASSERT_OP(labeling.shape(0),==,features.shape(0));
        CGP_ASSERT_OP(labeling.shape(0),==,mergedFeatures.shape(0));
        CGP_ASSERT_OP(features.shape(1),==,mergedFeatures.shape(1));
        CGP_ASSERT_OP(weightBuffer.shape(0),==,features.shape(0));

        // initalize with zeros
        std::fill(mergedFeatures.begin(),mergedFeatures.begin()+numberOfLabels,0.0);
        std::fill(weightBuffer.begin(),weightBuffer.begin()+numberOfLabels,0.0);


        const size_t nItems = labeling.shape(0);
        const size_t nFeatures = features.shape(1);

        // accumulate
        for(size_t i=0;i<nItems;++i){

            // get the label
            const size_t label = labeling(i);
            // get the weight
            const float weight = weights(i);
            // accumulate features
            for(size_t f=0;f<nFeatures;++f){
                mergedFeatures(label,f)+=weights*features(i,f);
            }
            // accumulate weights
            weightBuffer(i)+=weight;
        }
        // normalize 

        for(size_t label=0;label<numberOfLabels;++label){
            const float weight=weights(label);
            for(size_t f=0;f<nFeatures;++f){
                mergedFeatures(label,f)/=weight;
            }
        }

    }





    void clusterObjective(
        Cgp<CoordinateType,LabelType> & cgp,
        vigra::NumpyArray<1,LabelType>          labeling,
        vigra::NumpyArray<2,float>              features,
        vigra::NumpyArray<2,float>              mergedFeatures
    ){
        
    }



    void export_merge(){

        python::def("_mergeFeatures",vigra::registerConverters(&mergeFeatures),
            (
                python::arg("labeling"),
                python::arg("features"),
                python::arg("mergedFeatures")=python::object()
            )
        );

    }

}