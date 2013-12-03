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
#include "seglib/distances/distance.hxx"


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
                mergedFeatures(label,f)+=weight*features(i,f);
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


    float withinClusterDist(
        vigra::NumpyArray<1,LabelType>          labeling,
        const size_t                            numberOfLabels,
        vigra::NumpyArray<1,float>              weights,
        vigra::NumpyArray<2,float>              features,
        vigra::NumpyArray<2,float>              mergedFeatures
    ){
        CGP_ASSERT_OP(labeling.shape(0),==,features.shape(0));
        CGP_ASSERT_OP(labeling.shape(0),==,mergedFeatures.shape(0));
        CGP_ASSERT_OP(features.shape(1),==,mergedFeatures.shape(1));


        const size_t nItems = labeling.shape(0);
        const size_t nFeatures = features.shape(1);

        float totalD = 0.0;

        for(size_t i=0;i<nItems;++i){

            // get the label
            const size_t label = labeling(i);

            // compute the distance between the feature of the item (superpixel)
            // and the feature of the cluster (cluster of superpixel / "HyperRegion")
            vigra::MultiArrayView<1,float> a=features.bindInner(i);
            vigra::MultiArrayView<1,float> b=mergedFeatures.bindInner(label);
            const  float d = distances::Distance<float>::klDivergenz(a.begin(),a.end(),b.begin(),b.end());
            // get the weight of the item (superpixel size)
            const float weight=weights(i);

            totalD += weight*d;


            //weightBuffer(l)+=weight
        }
        return totalD;
    }

    float betweenClusterDist(
        const Cgp<CoordinateType,LabelType> &   cgp,
        vigra::NumpyArray<1,LabelType>          labeling,
        const size_t                            numberOfLabels,
        vigra::NumpyArray<2,float>              mergedFeatures
    ){

        float totalD=0.0f;

        std::set<size_t> used;
        const size_t nBoundaries = cgp.numCells(1);
        for(size_t b=0;b<nBoundaries;++b){
            const size_t r1 = cgp.bound<1>(b,0)-1;
            const size_t r2 = cgp.bound<1>(b,1)-1; 

            size_t l1 =labeling(r1);
            size_t l2 =labeling(r2);

            // active boundarie ? 
            if(l1!=l2){

                if(l2<l1){
                    std::swap(l1,l2);
                }
                const size_t key = l1+ l2*numberOfLabels;
                if(used.find(key)==used.end()){
                    used.insert(key);
                    vigra::MultiArrayView<1,float> a=mergedFeatures.bindInner(l1);
                    vigra::MultiArrayView<1,float> b=mergedFeatures.bindInner(l2);
                    const float d = distances::Distance<float>::klDivergenz(a.begin(),a.end(),b.begin(),b.end());
                    totalD+=d;
                }
            }       
        }
        return totalD;
    }



    void export_merge(){

        python::def("_mergeFeatures",vigra::registerConverters(&mergeFeatures),
            (
                python::arg("labeling"),
                python::arg("numberOfLabels"),
                python::arg("weights"),
                python::arg("features"),
                python::arg("mergedFeatures"),
                python::arg("weightBuffer")
            )
        );

        python::def("_withinClusterDist",vigra::registerConverters(&withinClusterDist),
            (
                python::arg("labeling"),
                python::arg("numberOfLabels"),
                python::arg("weights"),
                python::arg("features"),
                python::arg("mergedFeatures")
            )
        );

        python::def("_betweenClusterDist",vigra::registerConverters(&betweenClusterDist),
            (
                python::arg("cgp"),
                python::arg("labeling"),
                python::arg("numberOfLabels"),
                python::arg("mergedFeatures")
            )
        );

    }

}