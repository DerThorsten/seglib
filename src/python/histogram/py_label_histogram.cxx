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




    vigra::NumpyAnyArray labelHistogram(
        vigra::NumpyArray<3, vigra::Multiband<LabelType>  >         img,
        const LabelType                                             nLabels,
        const size_t                                                r,
        //output
        vigra::NumpyArray<4, float >    res = vigra::NumpyArray<4, float >()
    ){ 
        const size_t nChannels=img.shape(2);
        // allocate output
        typedef typename vigra::NumpyArray<4, float >::difference_type Shape4;
        Shape4 shape(img.shape(0),img.shape(1),nChannels,nLabels);
        res.reshapeIfEmpty(shape);
        std::fill(res.begin(),res.end(),0.0);


        // coordinate in the res array (pixel wise histogram)
        // (x,y,c,bin)
        Shape4 histCoord;
        const vigra::TinyVector<float, 2>  radius1(r+1,r+1);
  
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

                    const LabelType label = img(c[0],c[1],histCoord[2]);

                   
                    histCoord[3] = label;

                    /*
                    std::cout<<"\n\n” channel "<<histCoord[2]<<"\n";
                    std::cout<<"value "<< value<<"\n";
                    std::cout<<"mi " << min(histCoord[2])<<"\n";
                    std::cout<<"ma " << max(histCoord[2])<<"\n";
                    std::cout<<"fa " << fac[histCoord[2]]<<"\n";
                    */

                    PHIST_ASSERT_OP(histCoord[3],<,nLabels);
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
            for(histCoord[3]=0;histCoord[3]<nLabels;++histCoord[3]){
                sum+=res(histCoord[0],histCoord[1],histCoord[2],histCoord[3]);
            }
            for(histCoord[3]=0;histCoord[3]<nLabels;++histCoord[3]){
                res(histCoord[0],histCoord[1],histCoord[2],histCoord[3])/=sum;
            }
        }
        return res;
    }



    vigra::NumpyAnyArray labelSimHistogram(
        vigra::NumpyArray<3, vigra::Multiband<LabelType>  >         img,
        vigra::NumpyArray<3, float  >                               labelSim,
        const LabelType                                             nLabels,
        const size_t                                                r,
        //output
        vigra::NumpyArray<4, float >    res = vigra::NumpyArray<4, float >()
    ){ 
        const size_t nChannels=img.shape(2);
        // allocate output
        typedef typename vigra::NumpyArray<4, float >::difference_type Shape4;
        Shape4 shape(img.shape(0),img.shape(1),nChannels,nLabels);
        res.reshapeIfEmpty(shape);
        std::fill(res.begin(),res.end(),0.0);

        PHIST_ASSERT_OP(labelSim.shape(0),==,labelSim.shape(1));
        PHIST_ASSERT_OP(labelSim.shape(0),==,nLabels);
        PHIST_ASSERT_OP(labelSim.shape(2),==,img.shape(2));
        // coordinate in the res array (pixel wise histogram)
        // (x,y,c,bin)
        Shape4 histCoord;
        const vigra::TinyVector<float, 2>  radius1(r+1,r+1);
  
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

                    const LabelType label = img(c[0],c[1],histCoord[2]);

                   
                    histCoord[3] = label;

                    /*
                    std::cout<<"\n\n” channel "<<histCoord[2]<<"\n";
                    std::cout<<"value "<< value<<"\n";
                    std::cout<<"mi " << min(histCoord[2])<<"\n";
                    std::cout<<"ma " << max(histCoord[2])<<"\n";
                    std::cout<<"fa " << fac[histCoord[2]]<<"\n";
                    */

                    PHIST_ASSERT_OP(histCoord[3],<,nLabels);
                    // increment hist



                    for(size_t ll = 0 ;ll<nLabels;++ll){
                        const float sim = labelSim(label,ll,histCoord[2]);
                        res(histCoord[0],histCoord[1],histCoord[2],ll)+=sim;
                    }

                    
                }
            }
        }

        // normalize

        for(histCoord[0]=0;histCoord[0]<img.shape(0);++histCoord[0])
        for(histCoord[1]=0;histCoord[1]<img.shape(1);++histCoord[1])
        for(histCoord[2]=0;histCoord[2]<img.shape(2);++histCoord[2]){

            float sum=0.0;
            for(histCoord[3]=0;histCoord[3]<nLabels;++histCoord[3]){
                sum+=res(histCoord[0],histCoord[1],histCoord[2],histCoord[3]);
            }
            for(histCoord[3]=0;histCoord[3]<nLabels;++histCoord[3]){
                res(histCoord[0],histCoord[1],histCoord[2],histCoord[3])/=sum;
            }
        }
        return res;
    }





    void moveMe(
        vigra::NumpyArray<2, float  >           globalFeatures,
        vigra::NumpyArray<1, vigra::Int64  >    batchIndex,
        vigra::NumpyArray<1, vigra::Int64  >    minCenterIndex,         
        vigra::NumpyArray<1, float  >           centerCount,   
        vigra::NumpyArray<2, float  >           centers
    ){

        const size_t batchSize = batchIndex.shape(0);
        const size_t nFeatures = globalFeatures.shape(1);
        for(size_t bi=0;bi<batchSize;++bi){
            const size_t ci = minCenterIndex(bi);
            centerCount(ci)+=1.0;
            const float rate = 1.0/centerCount(ci);
            for(size_t f=0;f<nFeatures;++f){
                centers(ci,f)*=(1.0-rate);
                centers(ci,f)+=rate*globalFeatures(batchIndex(bi),f);
            }
        }
    }




    void histDist(
        vigra::NumpyArray<2, float  > samples,
        vigra::NumpyArray<2, float  > centers,
        vigra::NumpyArray<2, float  > distances,
        const std::string distType 
    ){

        // renormalize centers 




        const size_t nSamples  = samples.shape(0);
        const size_t nFeatures = samples.shape(1);
        const size_t nCenters  = centers.shape(0);
        



        if(distType==std::string("bhattacharyya")){

            // bhattacharyya dist may shift centers
            for(size_t c=0;c<nCenters;++c){

                float minV=std::numeric_limits<float>::infinity();
                for(size_t f=0;f<nFeatures;++f){
                    minV=std::min(minV,centers(c,f));
                }
                if (minV<0.000001f){
                    for(size_t f=0;f<nFeatures;++f){
                        centers(c,f)-=minV;
                    }
                }
                float sum=0;
                for(size_t f=0;f<nFeatures;++f){
                    sum+=centers(c,f);
                }
                if(sum>0.999999){
                    for(size_t f=0;f<nFeatures;++f){
                        centers(c,f)/=sum;
                    }
                }

            }


            for(size_t s=0;s<nSamples;++s)
            for(size_t c=0;c<nCenters;++c){
                // compute distace

                float sum=0.0;
                for(size_t f=0;f<nFeatures;++f){
                    

                    const float sv=samples(s,f);
                    const float cv=samples(c,f);
                    PHIST_ASSERT_OP(sv , > , -0.0000001);
                    PHIST_ASSERT_OP(cv , > , -0.0000001);
                    PHIST_ASSERT_OP(sv , < , 1.0000001);
                    PHIST_ASSERT_OP(cv , < , 1.0000001);
                    sum+=std::sqrt(samples(s,f)*centers(c,f));
                }
                PHIST_ASSERT_OP(sum , <= , 1.0001);
                sum=std::min(sum,0.9999999f);
                distances(s,c)=(1.0f - sum);
                //std::cout<<"distances "<<distances(s,c)<<"\n";
            }
        }
        else if(distType==std::string("chi2")){
            for(size_t s=0;s<nSamples;++s)
            for(size_t c=0;c<nCenters;++c){
                // compute distace

                float sum=0.0;
                for(size_t f=0;f<nFeatures;++f){
                    const float sv=samples(s,f);
                    const float cv=samples(c,f);

                    sum+=std::pow(sv-cv,2)/(sv+cv);
                }
                distances(s,c)=0.5*sum;
                //std::cout<<"distances "<<distances(s,c)<<"\n";
            }
        }
        else{
            PHIST_ASSERT_OP(0,==,1);
        }
    }



    void export_label_histogram(){



        python::def("_label_histogram_",vigra::registerConverters(&labelHistogram),
            (
                python::arg("img"),
                python::arg("nLabels"),
                python::arg("r"),
                python::arg("out")=python::object()
            )
        );

        python::def("_label_sim_histogram_",vigra::registerConverters(&labelSimHistogram),
            (
                python::arg("img"),
                python::arg("labelSim"),
                python::arg("nLabels"),
                python::arg("r"),
                python::arg("out")=python::object()
            )
        );


        python::def("moveMe",vigra::registerConverters(&moveMe),
            (
                python::arg("globalFeatures"),
                python::arg("batchIndex"),
                python::arg("minCenterIndex"),
                python::arg("centerCount"),
                python::arg("centers")
            )
        );

        python::def("histDist",vigra::registerConverters(&histDist),
            (
                python::arg("samples"),
                python::arg("centers"),
                python::arg("distances"),
                python::arg("distType")=std::string("bhattacharyya")
            )
        );

    }

}