#define PY_ARRAY_UNIQUE_SYMBOL phist_PyArray_API
#define NO_IMPORT_ARRAY

#include <string>
#include <cmath>

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>


#include <boost/array.hpp>



#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>

#include "seglib/histogram/histogram.hxx"
#include "seglib/histogram/histogram_python.hxx"


namespace python = boost::python;

namespace histogram {




    vigra::NumpyAnyArray jointColorHistogram(

        vigra::NumpyArray<2,  vigra::TinyVector<float,3>  >    img,
        const vigra::TinyVector<float, 3> &                    min,
        const vigra::TinyVector<float, 3> &                    max,
        const vigra::TinyVector<float, 3> &                    bins,
        const size_t                                           r,
        //output
        vigra::NumpyArray<5, float >                           res = vigra::NumpyArray<5, float >()


    ){

        // allocate output
        typedef typename vigra::NumpyArray<5, float >::difference_type Shape5;
        Shape5 shape(img.shape(0),img.shape(1),int(bins[0]),int(bins[1]),int(bins[2]));
        res.reshapeIfEmpty(shape);
        std::fill(res.begin(),res.end(),0.0);
        // 
        Shape5 histCoord;
        const vigra::TinyVector<uint,  3>  ones(1,1,1);
        const vigra::TinyVector<float, 2>  radius(r,r);
        const vigra::TinyVector<float, 2>  radius1(r+1,r+1);
        const vigra::TinyVector<float, 3>  fac = ( (bins-ones) / (max-min) );


        vigra::TinyVector<int,2>  start,end,c;

        for(histCoord[0]=0;histCoord[0]<img.shape(0);++histCoord[0])
        for(histCoord[1]=0;histCoord[1]<img.shape(1);++histCoord[1]){


            for(int d=0;d<2;++d){
                start[d]   = std::max(int(0),            int(histCoord[d]) - int(r));
                end[d]     = std::min(int(img.shape(d)), int(histCoord[d] + (r+1) )); 
            }


            for(c[0]=start[0];c[0]<end[0];++c[0])
            for(c[1]=start[1];c[1]<end[1];++c[1]){

                
                // get the pixel value at c
                vigra::TinyVector<float, 3>  pixelValue = img(c[0],c[1]);
                pixelValue -= min;
                pixelValue *= fac;

                // (the first two coordinates of hist coord are filled)
                histCoord[2]=int(pixelValue[0]);
                histCoord[3]=int(pixelValue[1]);
                histCoord[4]=int(pixelValue[2]);


                PHIST_ASSERT_OP(histCoord[0],<,img.shape(0));
                PHIST_ASSERT_OP(histCoord[1],<,img.shape(1));

                PHIST_ASSERT_OP(histCoord[2],<,bins[0]);
                PHIST_ASSERT_OP(histCoord[3],<,bins[1]);
                PHIST_ASSERT_OP(histCoord[4],<,bins[2]);

                // increment counter
                res(histCoord[0],histCoord[1],histCoord[2],histCoord[3],histCoord[4])+=1.0;
            }   
        }   

        // normalizes
        for(histCoord[0]=0;histCoord[0]<img.shape(0);++histCoord[0])
        for(histCoord[1]=0;histCoord[1]<img.shape(1);++histCoord[1]){

            float sum=0;

            for(histCoord[2]=0;histCoord[2]<res.shape(2);++histCoord[2])
            for(histCoord[3]=0;histCoord[3]<res.shape(3);++histCoord[3])
            for(histCoord[4]=0;histCoord[4]<res.shape(4);++histCoord[4]){

                sum+=res(histCoord[0],histCoord[1],histCoord[2],histCoord[3],histCoord[4]);
            }
            for(histCoord[2]=0;histCoord[2]<res.shape(2);++histCoord[2])
            for(histCoord[3]=0;histCoord[3]<res.shape(3);++histCoord[3])
            for(histCoord[4]=0;histCoord[4]<res.shape(4);++histCoord[4]){

                res(histCoord[0],histCoord[1],histCoord[2],histCoord[3],histCoord[4])/=sum;
            }
        }



        return res;
    }



    vigra::NumpyAnyArray batchHistogram(
        vigra::NumpyArray<3, vigra::Multiband<float>  >   img,
        vigra::NumpyArray<1, float  >   min,
        vigra::NumpyArray<1, float  >   max,
        const size_t                    bins,
        const size_t                    r,
        //output
        vigra::NumpyArray<4, float >    res = vigra::NumpyArray<4, float >()
    ){ 
        const size_t nChannels=img.shape(2);
        // allocate output
        typedef typename vigra::NumpyArray<4, float >::difference_type Shape4;
        Shape4 shape(img.shape(0),img.shape(1),nChannels,bins);
        res.reshapeIfEmpty(shape);
        std::fill(res.begin(),res.end(),0.0);


        // coordinate in the res array (pixel wise histogram)
        // (x,y,c,bin)
        Shape4 histCoord;
        const vigra::TinyVector<float, 2>  radius1(r+1,r+1);
        // channel wise factor
        std::vector<float> fac(nChannels);
        for(size_t channel=0;channel<nChannels;++channel){
            fac[channel]= float(bins-1) / (max(channel)-min(channel)); 
        }


        vigra::TinyVector<int,2>  start,end,c;

        for(histCoord[0]=0;histCoord[0]<img.shape(0);++histCoord[0])
        for(histCoord[1]=0;histCoord[1]<img.shape(1);++histCoord[1]){


            for(int d=0;d<2;++d){
                start[d]   = std::max(int(0),            int(histCoord[d]) - int(r));
                end[d]     = std::min(int(img.shape(d)), int(histCoord[d] + (r+1) )); 
            }


            for(c[0]=start[0];c[0]<end[0];++c[0])
            for(c[1]=start[1];c[1]<end[1];++c[1]){

                // iterate over all channels
                for(histCoord[2]=0;histCoord[2]<nChannels;++histCoord[2] ){

                    const float value = img(c[0],c[1],histCoord[2]);

                   


                    histCoord[3] = static_cast<int>((value - min(histCoord[2]) )*fac[histCoord[2]]);

                    /*
                    std::cout<<"\n\n” channel "<<histCoord[2]<<"\n";
                    std::cout<<"value "<< value<<"\n";
                    std::cout<<"mi " << min(histCoord[2])<<"\n";
                    std::cout<<"ma " << max(histCoord[2])<<"\n";
                    std::cout<<"fa " << fac[histCoord[2]]<<"\n";
                    */

                    PHIST_ASSERT_OP(histCoord[3],<,bins);
                    // increment hist
                    res(histCoord[0],histCoord[1],histCoord[2],histCoord[3])+=1.0;
                }
            }
        }

        // normalize

        for(histCoord[0]=0;histCoord[0]<img.shape(0);++histCoord[0])
        for(histCoord[1]=0;histCoord[1]<img.shape(1);++histCoord[1])
        for(histCoord[2]=0;histCoord[2]<img.shape(2);++histCoord[2]){

            float sum=0.0;
            for(histCoord[3]=0;histCoord[3]<bins;++histCoord[3]){
                sum+=res(histCoord[0],histCoord[1],histCoord[2],histCoord[3]);
            }
            for(histCoord[3]=0;histCoord[3]<bins;++histCoord[3]){
                res(histCoord[0],histCoord[1],histCoord[2],histCoord[3])/=sum;
            }
        }
        return res;
    }


    /*

    vigra::NumpyAnyArray labelMaskedHistogram(
        // input data (for example an rgb image with xyc order)
        vigra::NumpyArray<3, vigra::Multiband<float>  >         img,
        // input labeling for which are used as mask images
        vigra::NumpyArray<3, vigra::Multiband<vigra::UInt64> >  labelings,
        vigra::NumpyArray<1, UInt64>                            numberOfLabels,
        vigra::NumpyArray<1, float>                             weights
    ){

        const size_t numberOfInputLabelings = labelings.shape(2);
        
    }
    */


    void export_histogram(){

        python::def("_jointColorHistogram_",vigra::registerConverters(&jointColorHistogram),
            (
                python::arg("img"),
                python::arg("dmin"),
                python::arg("dmax"),
                python::arg("bins"),
                python::arg("r"),
                python::arg("out")=python::object()
            )
        );

        python::def("_batchHistogram_",vigra::registerConverters(&batchHistogram),
            (
                python::arg("img"),
                python::arg("dmin"),
                python::arg("dmax"),
                python::arg("bins"),
                python::arg("r"),
                python::arg("out")=python::object()
            )
        );

    }

}