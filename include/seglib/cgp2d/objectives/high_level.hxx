#ifndef CGP_HIGH_LEVEL
#define CGP_HIGH_LEVEL


#include <seglib/cgp2d/cgp2d.hxx>
#include <vigra/multi_array.hxx>
#include "seglib/distances/distance.hxx"


namespace cgp2d {

    // tgrid and input image type
    typedef Cgp<CoordinateType,LabelType> CgpType;
    typedef CgpType::TopologicalGridType TopologicalGridType;



    template<class CGP,class T>
    class HighLevelObjective{   
    private:
        typedef typename vigra::MultiArrayView<1,LabelType>::difference_type Shape1;
    public:

        HighLevelObjective(const CGP & cgp)
        :   cgp_(cgp),
            cell2Features_(),
            mergedCell2Features_(),
            primalLabels_(Shape1(cgp.numCells(2))),
            primalLabelsWeight_(Shape1(cgp.numCells(2))){
        }
        
        void setRegionFeatures(const vigra::MultiArrayView<2,T> & other){
            cell2Features_ = other;
            mergedCell2Features_.reshape(cell2Features_.shape());
            std::cout<<"shape "<<cell2Features_.shape(0)<<" "<<cell2Features_.shape(1)<<"\n";
            std::cout<<"shape "<<mergedCell2Features_.shape(0)<<" "<<mergedCell2Features_.shape(1)<<"\n";

            
        }

        void setPrimalLabels(const vigra::MultiArrayView<1,LabelType> & labels){
            primalLabels_=labels;
            nCC_ = *std::max_element(primalLabels_.begin(),primalLabels_.end())+1;

        }

        void mergeFeatures();




        // 
        void writeBackMergedFeatures(vigra::MultiArrayView<2,T> & cell2Features)const{
            CGP_ASSERT_OP(cell2Features.shape(0),==,cell2Features_.shape(0));

            const size_t nReg  = cgp_.numCells(2);
            const size_t nFeat = cell2Features_.shape(1);
            CGP_ASSERT_OP(nReg,==,cell2Features_.shape(0));

            // accumulation
            for(size_t r=0;r<nReg;++r){
                const size_t l = primalLabels_(r);
                const float weight = static_cast<float>(cgp_. template cellSizeT<2>(r));
                for(size_t f=0;f<nFeat;++f){
                    cell2Features(r,f)=mergedCell2Features_(l,f);
                }
            }
        }




        float withinClusterDistance();

        template<class DIST_FUNCTOR>
        float betweenClusterDistance(DIST_FUNCTOR & distFunctor, const double gamma);

        const vigra::MultiArrayView<2,T> & features()const{
            return cell2Features_;
        }

    private:    

        const CGP & cgp_;

        vigra::MultiArrayView<2,T>      cell2Features_;
        vigra::MultiArray<2,T>          mergedCell2Features_;
        vigra::MultiArray<1,LabelType>  primalLabels_;
        vigra::MultiArray<1,float>      primalLabelsWeight_;
        size_t nCC_;
    };  




    template<class CGP,class T>
    void HighLevelObjective<CGP,T>::mergeFeatures(){





        mergedCell2Features_=0.0;
        primalLabelsWeight_ =0.0;


        const size_t nReg  = cgp_.numCells(2);
        const size_t nFeat = cell2Features_.shape(1);
        CGP_ASSERT_OP(nReg,==,cell2Features_.shape(0));

        // accumulation
        for(size_t r=0;r<nReg;++r){
            const size_t l = primalLabels_(r);
            const float weight = static_cast<float>(cgp_. template cellSizeT<2>(r));
            primalLabelsWeight_(l)+= weight;

            for(size_t f=0;f<nFeat;++f){
                mergedCell2Features_(l,f)+=weight*cell2Features_(r,f);
            }
        }
        // normalization
        for(size_t l=0;l<nCC_;++l){
            const float weight = primalLabelsWeight_(l);
            for(size_t f=0;f<nFeat;++f){
                mergedCell2Features_(l,f)/=weight;
                //std::cout<<"mergedCell2Features_ "<<mergedCell2Features_(l,f)<<"\n";
            }
        }
    }

    template<class CGP,class T>
    float HighLevelObjective<CGP,T>::withinClusterDistance(){

        //std::cout<<"\n\n\n computing withing cluster dist with "<<nCC_<<"clusters\n";
        dist::ChiSquared<double> distFunctor;
        double totalD=0.0;
        double weightSum=0.0;
        const size_t nReg  = cgp_.numCells(2);
        const size_t nFeat = cell2Features_.shape(1);
        CGP_ASSERT_OP(nReg,==,cell2Features_.shape(0));


        for(size_t r=0;r<nReg;++r){
            const size_t l = primalLabels_(r);
            vigra::MultiArrayView<1,float> sp =cell2Features_.bindInner(r);
            vigra::MultiArrayView<1,float> reg=mergedCell2Features_.bindInner(l);

            const double d=distFunctor(sp,reg);
            const double weight = static_cast<double>(cgp_. template cellSizeT<2>(r));
            //std::cout<<"distance   "<<d<<"\n";
            totalD+=d*weight;
            weightSum+=weight;
        }
        return static_cast<float>(totalD);

    }

    template<class CGP,class T>
    template<class DIST_FUNCTOR>
    float HighLevelObjective<CGP,T>::betweenClusterDistance(DIST_FUNCTOR & distFunctor,const double gamma){
            
        std::set<size_t> used;
        double totalD=0.0;
        double weightSum=0.0;

        for(size_t i=0;i<cgp_.numCells(1);++i){
            const size_t r1=cgp_. template bound<1> (i,0)-1;
            const size_t r2=cgp_. template bound<1> (i,1)-1;
            size_t l1=primalLabels_(r1);
            size_t l2=primalLabels_(r2);

            if(l1!=l2){
                if(l2<l1){
                    std::swap(l1,l2);
                } 
                CGP_ASSERT_OP(l1,<,nCC_);
                const size_t key = l1 + l2*nCC_;
                if(used.find(key)==used.end()){
                    used.insert(key);

                    vigra::MultiArrayView<1,float> c1=mergedCell2Features_.bindInner(l1);
                    vigra::MultiArrayView<1,float> c2=mergedCell2Features_.bindInner(l2);

                    CGP_ASSERT_OP(c1.shape(0),==,cell2Features_.shape(2));

                    const double d=distFunctor(c1,c2);

                    // from distance to energy
                    const double e1     = std::exp(-1.0*gamma*d);
                    const double e0     = 1.0-e1;
                    const double weight = e1-e0;


                    //std::cout<<"d   "<<d <<"\n";
                    //std::cout<<"e0  "<<e0<<"\n";
                    //std::cout<<"e1  "<<e1<<"\n";
                    //std::cout<<"w   "<<weight<<"\n\n";



                    totalD+=weight;
                }
            }
        }

        return static_cast<float>(totalD);
    }

} // end namespace cgp2d

#endif //PHIST_PHIST_HXX