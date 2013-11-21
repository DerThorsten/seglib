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




    vigra::NumpyAnyArray  regionAffinity(
        vigra::NumpyArray<2, vigra::Multiband<vigra::UInt64>  >   labelings,
        vigra::NumpyArray<1, vigra::UInt64>                       nLabels,                      
        //output
        vigra::NumpyArray<2, float >    res = vigra::NumpyArray<4, float >()
    ){ 
        const size_t nSeg = labelings.shape(1);
        const size_t nReg = labelings.shape(1);
        // allocate output
        typedef typename vigra::NumpyArray<2, float >::difference_type Shape2;
        Shape2 shape(nReg,nReg);
        res.reshapeIfEmpty(shape);
        std::fill(res.begin(),res.end(),0.0);


        for(size_t s=0;s<nSeg;++s){

            const size_t nL=nLabels(s);

            std::vector< std::vector< vigra::UInt64> > regWithLabel;
            for(size_t r=0;r<nReg;++r){

                const vigra::UInt64 label=labelings(r,s);
                regWithLabel[label].push_back(r);
            }

            for(size_t l=0;l<nL;++l){

                for(size_t r0=0;    r0<regWithLabel[l].size()-1 ;++r0)
                for(size_t r1=r0+1; r1<regWithLabel[l].size()   ;++r1){
                    res[r0,r1]+=1.0;
                    res[r1,r0]+=1.0;
                }
            }
        }
        return res;
    }



    void export_segcompare(){

        python::def("_regionAffinity",vigra::registerConverters(&regionAffinity),
            (
                python::arg("labelings"),
                python::arg("nLabels"),
                python::arg("out")=python::object()
            )
        );

    }

}