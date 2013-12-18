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

namespace cgp2d
{


template <class RAIter, class Compare>
void argsort(RAIter iterBegin, RAIter iterEnd, Compare comp, 
    std::vector<size_t>& indexes) {

    std::vector< std::pair<size_t,RAIter> > pv ;
    pv.reserve(iterEnd - iterBegin) ;

    RAIter iter ;
    size_t k ;
    for (iter = iterBegin, k = 0 ; iter != iterEnd ; iter++, k++) {
        pv.push_back( std::pair<int,RAIter>(k,iter) ) ;
    }

    std::sort(pv.begin(), pv.end(), 
        [&comp](const std::pair<size_t,RAIter>& a, const std::pair<size_t,RAIter>& b) -> bool 
        { return comp(*a.second, *b.second) ; }) ;

    indexes.resize(pv.size()) ;
    std::transform(pv.begin(), pv.end(), indexes.begin(), 
        [](const std::pair<size_t,RAIter>& a) -> size_t { return a.first ; }) ;
}

//centerCoordinate
template<class BB>
float bbSize(const BB & bb){
    float sx = std::abs(bb.first[0]-bb.second[0]+1);
    float sy = std::abs(bb.first[0]-bb.second[0]+1);
    sx = (sx+1)/2.0;
    sy = (sy+1)/2.0;

    return std::sqrt(sx*sx + sy*sy);
}









template<class CGP>
vigra::NumpyAnyArray  countMultiEdges(
    const CGP & cgp
){

    typedef vigra::NumpyArray<1,float>  ResultArray;
    typedef typename ResultArray::difference_type ShapeType;
    typedef typename CGP::Cells1 Cells1;
    typedef typename CGP::Cell1  Cell1;
    typedef typename CGP::Cells2 Cells2;
    typedef typename CGP::Cell2  Cell2;


    typedef typename CGP::CellAdjacencyGraphVectorType CellAdjGraph;



    const size_t numBoundaries = cgp.numCells(1);
    const size_t numRegions    = cgp.numCells(2);
    const Cells1 & cells1=cgp.geometry1();
    const Cells2 & cells2=cgp.geometry2();
    // initialize result array
    ResultArray resultArray = ResultArray(ShapeType(numBoundaries));


    // Adj Graph 
    CellAdjGraph cell1AdjGraph;
    CellAdjGraph cell2AdjGraph;

    // fill adjacency graph
    cgp.cellAdjacencyGraphVector(1,cell1AdjGraph);
    cgp.cellAdjacencyGraphVector(2,cell2AdjGraph);

    typedef unsigned long long  KeyType;
    typedef std::map<KeyType, size_t> MapType;
    typedef typename MapType::const_iterator MapIter;


    MapType counter;

    for(size_t bi=0;bi<numBoundaries;++bi){
        const Cell1 & cell1  = cells1[bi];
        const KeyType ri0     = static_cast<KeyType>(cell1.bounds()[0]-1);
        const KeyType ri1     = static_cast<KeyType>(cell1.bounds()[1]-1);
        const KeyType key = std::min(ri0,ri1) + std::max(ri0,ri1)*numRegions;
        counter[key]=0;
    }
    for(size_t bi=0;bi<numBoundaries;++bi){
        const Cell1 & cell1  = cells1[bi];
        const KeyType ri0     = static_cast<KeyType>(cell1.bounds()[0]-1);
        const KeyType ri1     = static_cast<KeyType>(cell1.bounds()[1]-1);
        const KeyType key = std::min(ri0,ri1) + std::max(ri0,ri1)*numRegions;
        counter[key]=counter[key]+1;
    }
    for(size_t bi=0;bi<numBoundaries;++bi){
        const Cell1 & cell1  = cells1[bi];
        const KeyType ri0     = static_cast<KeyType>(cell1.bounds()[0]-1);
        const KeyType ri1     = static_cast<KeyType>(cell1.bounds()[1]-1);
        const KeyType key = std::min(ri0,ri1) + std::max(ri0,ri1)*numRegions;
        resultArray(bi)=static_cast<float>(counter[key]);
    }
    return resultArray;
}



template<class CGP>
vigra::NumpyAnyArray  cell1TopoFeatures(
    const CGP & cgp
){

    typedef vigra::NumpyArray<2,float>  ResultArray;
    typedef typename ResultArray::difference_type ShapeType;
    typedef typename CGP::Cells1 Cells1;
    typedef typename CGP::Cell1  Cell1;
    typedef typename CGP::Cells2 Cells2;
    typedef typename CGP::Cell2  Cell2;


    typedef typename CGP::CellAdjacencyGraphVectorType CellAdjGraph;



    const size_t numBoundaries = cgp.numCells(1);
    const Cells1 & cells1=cgp.geometry1();
    const Cells2 & cells2=cgp.geometry2();
    // initialize result array
    ResultArray resultArray = ResultArray(ShapeType(numBoundaries,5));


    // Adj Graph 
    CellAdjGraph cell1AdjGraph;
    CellAdjGraph cell2AdjGraph;

    // fill adjacency graph
    cgp.cellAdjacencyGraphVector(1,cell1AdjGraph);
    cgp.cellAdjacencyGraphVector(2,cell2AdjGraph);

    for(size_t bi=0;bi<numBoundaries;++bi){

        const Cell1 & cell1  = cells1[bi];

        {   // from boundary itself
            resultArray(bi,0) = cell1AdjGraph[bi].size();
        }

        {  // from the 2 adj regions

            const size_t ri0     = cell1.bounds()[0]-1;
            const size_t ri1     = cell1.bounds()[1]-1;
            const Cell2 & cell20 = cells2[ri0];
            const Cell2 & cell21 = cells2[ri1];
            const float nAdj0    = cell2AdjGraph[ri0].size();
            const float nAdj1    = cell2AdjGraph[ri1].size();

            resultArray(bi,1) =  (nAdj0+nAdj1)/2.0;
            resultArray(bi,2) =  std::abs(nAdj0-nAdj1)/2.0;
            resultArray(bi,3) =  std::min(nAdj0,nAdj1);
            resultArray(bi,4) =  std::max(nAdj0,nAdj1);
        }
    }
    return resultArray;
}


template<class CGP>
vigra::NumpyAnyArray  cell1GeoFeatures(
    const CGP & cgp
){

    typedef vigra::NumpyArray<2,float>  ResultArray;
    typedef typename ResultArray::difference_type ShapeType;
    typedef typename CGP::Cells1 Cells1;
    typedef typename CGP::Cell1  Cell1;
    typedef typename CGP::Cells2 Cells2;
    typedef typename CGP::Cell2  Cell2;
    typedef typename Cell1::FloatPointType FloatPointType;
    typedef typename Cell1::PointType PointType;
    typedef std::pair<PointType,PointType> BouningBoxType;

    const size_t numBoundaries = cgp.numCells(1);
    const Cells1 & cells1=cgp.geometry1();
    const Cells2 & cells2=cgp.geometry2();
    // initialize result array
    ResultArray resultArray = ResultArray(ShapeType(numBoundaries,13));
    std::fill(resultArray.begin(),resultArray.end(),0.0);



    for(size_t bi=0;bi<numBoundaries;++bi){

        const Cell1 & cell1  = cells1[bi];

        {   // from boundary itself

            const FloatPointType center = cell1.centerCoordinate();
            BouningBoxType bBox   = cell1.boundingBox();
            bBox.second[0]+=1;
            bBox.second[1]+=1;

            const float lineSize        = cell1.size();
            const float bbDiagonal      = vigra::norm(bBox.first -bBox.second );
            const float relBBoxSize     = bbDiagonal/lineSize;
            const float startEndDist    = vigra::norm(cell1[0]-cell1[lineSize-1]);
            const float relStartEndSize = startEndDist/lineSize;

            
            resultArray(bi,0)=lineSize;
            resultArray(bi,1)=bbDiagonal;
            resultArray(bi,2)=relBBoxSize;
            resultArray(bi,3)=startEndDist;
            resultArray(bi,4)=relStartEndSize;
        }

        {  // from the 2 adj regions

            const size_t ri0     = cell1.bounds()[0]-1;
            const size_t ri1     = cell1.bounds()[1]-1;
            const Cell2 & cell20 = cells2[ri0];
            const Cell2 & cell21 = cells2[ri1];
            BouningBoxType bBox0 = cell20.boundingBox();
            BouningBoxType bBox1 = cell21.boundingBox();
    
            const float bbSize0 = bbSize(bBox0);
            const float bbSize1 = bbSize(bBox1);

            const float size0 = cell20.size();
            const float size1 = cell21.size();

            const float relSize0 = bbSize0/size0;
            const float relSize1 = bbSize1/size1;


            resultArray(bi,5) = (size0 + size1)/2.0;
            resultArray(bi,6) = std::abs(size0 - size1);
            resultArray(bi,7) = std::min(size0 , size1);
            resultArray(bi,8) = std::max(size0 , size1);

            resultArray(bi,9) = (relSize0 +relSize1)/2.0;
            resultArray(bi,10) = std::abs(relSize0 -relSize1);
            resultArray(bi,11) = std::min(relSize0 , relSize1);
            resultArray(bi,12) = std::max(relSize0 , relSize1);
        }


    
    }
    return resultArray;
}





template<class CGP>
vigra::NumpyAnyArray  relativeCenterDist(
    const CGP & cgp
){

    typedef vigra::NumpyArray<1,float>  ResultArray;
    typedef typename ResultArray::difference_type ShapeType;
    typedef typename CGP::Cells1 Cells1;
    typedef typename CGP::Cell1 Cell1;
    typedef typename Cell1::FloatPointType FloatPointType;
    const size_t numBoundaries = cgp.numCells(1);
    const Cells1 & cells1=cgp.geometry1();

    // initialize result array
    ResultArray resultArray = ResultArray(ShapeType(numBoundaries));
    std::fill(resultArray.begin(),resultArray.end(),0.0);


    FloatPointType imageCenter(0.5,0.5);
    const float maxDist = vigra::norm(imageCenter - FloatPointType(1.0,1.0));


    for(size_t bi=0;bi<numBoundaries;++bi){
        const Cell1 & cell1   = cells1[bi];
        FloatPointType centerCoordinate = cell1.centerCoordinate();
        centerCoordinate[0]/=cgp.shape(0);
        centerCoordinate[1]/=cgp.shape(1);

        const float distance = vigra::norm(imageCenter -centerCoordinate );
        resultArray(bi)=distance;
        
    }
    return resultArray;
}


template<class CGP>
vigra::NumpyAnyArray  boarderTouch(
    const CGP & cgp
){

    typedef vigra::NumpyArray<1,float>  ResultArray;
    typedef typename ResultArray::difference_type ShapeType;
    typedef typename CGP::Cells1 Cells1;
    typedef typename CGP::Cell1 Cell1;

    const size_t numBoundaries = cgp.numCells(1);
    const Cells1 & cells1=cgp.geometry1();

    // initialize result array
    ResultArray resultArray = ResultArray(ShapeType(numBoundaries));
    std::fill(resultArray.begin(),resultArray.end(),0.0);


    const size_t maxX =cgp.shape(0)-1;
    const size_t maxY =cgp.shape(1)-1;

    for(size_t bi=0;bi<numBoundaries;++bi){
        
        float numBoarderTouch=0.0;

        const Cell1 & cell1   = cells1[bi];
        const size_t numCoord = cell1.size();


        const size_t cxStart = cell1[0][0];
        const size_t cyStart = cell1[0][1];

        const size_t cxEnd = cell1[numCoord-1][0];
        const size_t cyEnd = cell1[numCoord-1][1];

        if ( 
            ( cxStart == 0  || cxStart == maxX) || 
            ( cyStart == 0  || cyStart == maxY) 
        ){
            numBoarderTouch+=1;
        }

        if ( 
            ( cxEnd == 0  || cxEnd == maxX) || 
            ( cyEnd == 0  || cyEnd == maxY) 
        ){
            numBoarderTouch+=1;
        }
        resultArray(bi)=numBoarderTouch;
  
    }
    return resultArray;
}




template<class CGP>
vigra::NumpyAnyArray  cell1GraphStealing(
    const CGP & cgp,
    vigra::NumpyArray<1,float> cell1Features,
    const float frac,
    const float pl,
    const float ph
){

    typedef vigra::NumpyArray<1,float>  ResultArray;
    typedef typename ResultArray::difference_type ShapeType;
    typedef typename CGP::Cells0 Cells0;
    typedef typename CGP::Cells1 Cells1;
    typedef typename CGP::Cell0 Cell0;
    typedef typename CGP::Cell1 Cell1;
    typedef typename CGP::CellAdjacencyGraphVectorType CellAdjGraph;

    const size_t cellType = 1;
    const size_t numCells = cgp.numCells(cellType);

    using namespace boost::accumulators;
    typedef accumulator_set<double, stats<
        tag::min,
        tag::mean,
        tag::median(with_p_square_quantile),
        tag::extended_p_square_quantile
    > > AccSet;


    typedef accumulator_set<double, stats<tag::tail_quantile<right> > > accumulator_t_right;
    typedef accumulator_set<double, stats<tag::tail_quantile<left> > >  accumulator_t_left;



     // initialize result array
    ResultArray resultArray = ResultArray(ShapeType(numCells));

    std::copy(cell1Features.begin(),cell1Features.end(),resultArray.begin());

    // Adj Graph 
    CellAdjGraph cellAdjGraph;
    // fill adjacency graph
    cgp.cellAdjacencyGraphVector(cellType,cellAdjGraph);

    boost::array<double,2> probs = { pl,ph};

    for(size_t ci=0;ci<numCells;++ci){


        accumulator_t_right accR( right_tail_cache_size = 1000 );
        accumulator_t_left  accL( left_tail_cache_size = 1000 );
        AccSet accSet(extended_p_square_probabilities = probs);

        const double ownVal = static_cast<double>(cell1Features(ci));
        accSet(ownVal);

        const size_t nAdj=cellAdjGraph[ci].size();
        for(size_t n=0;n<nAdj;++n){

            const double val = cell1Features(cellAdjGraph[ci][n]-1);

            //std::cout<<"value "<<val<<"\n";
            accSet(val);
            accR(val);
            accL(val);
        }

        //const double ql = boost::accumulators::quantile(accSet, quantile_probability = pl);
        //const double qh = boost::accumulators::quantile(accSet, quantile_probability = ph);


        const double ql = boost::accumulators::quantile(accL, quantile_probability = pl);
        const double qh = boost::accumulators::quantile(accR, quantile_probability = ph);

        //std::cout<<"ql "<<ql<<"\n";
        //std::cout<<"qh "<<qh<<"\n";

        /*
        std::cout<<"left\n";
        std::cout<<"ql "<<boost::accumulators::quantile(accL, quantile_probability = pl)<<"\n";
        std::cout<<"qh "<<boost::accumulators::quantile(accL, quantile_probability = ph)<<"\n";

        std::cout<<"right\n";
        std::cout<<"ql "<<boost::accumulators::quantile(accR, quantile_probability = pl)<<"\n";
        std::cout<<"qh "<<boost::accumulators::quantile(accR, quantile_probability = ph)<<"\n\n";
        */
        size_t nl=0,nh=0;




        if(ownVal <= ql && nAdj!=0){

            for(size_t n=0;n<nAdj;++n){
                nl+=static_cast<double>(cell1Features(cellAdjGraph[ci][n]-1)) <= ql ? 1 : 0;
                nh+=static_cast<double>(cell1Features(cellAdjGraph[ci][n]-1)) >= qh ? 1 : 0;
            }
            CGP_ASSERT_OP(nh,>=,1);
            CGP_ASSERT_OP(nl,>=,1);

            const double forOthers = (ownVal*frac)/static_cast<double>(nh);
            resultArray(ci)-=(ownVal*frac);
            for(size_t n=0;n<nAdj;++n){
                if(static_cast<double>(cell1Features(cellAdjGraph[ci][n]-1)) >= qh){
                    resultArray(cellAdjGraph[ci][n]-1)+=forOthers;
                }
            }
        }
    }
    return resultArray;
}




template<class CGP>
vigra::NumpyAnyArray  graphBiMean(
    const CGP & cgp,
    vigra::NumpyArray<1,float> cell1Features,
    const float alpha,  // high alpha means a lot of smoothing (alpha in [0,1] )
    const float gamma   // LOW gamma means a lot of smoothing
){

    typedef vigra::NumpyArray<1,float>  ResultArray;
    typedef typename ResultArray::difference_type ShapeType;

    typedef typename CGP::Cells0 Cells0;
    typedef typename CGP::Cells1 Cells1;

    typedef typename CGP::Cell0 Cell0;
    typedef typename CGP::Cell1 Cell1;

    const size_t numJunctions  = cgp.numCells(0);
    const size_t numBoundaries = cgp.numCells(1);
    const Cells1 & cells1=cgp.geometry1();
    const Cells0 & cells0=cgp.geometry0();

    // initialize result array
    ResultArray resultArray = ResultArray(ShapeType(numBoundaries));
    std::copy(cell1Features.begin(),cell1Features.end(),resultArray.begin());


    for(size_t bi=0;bi<numBoundaries;++bi){
        const Cell1 & cell1 = cells1[bi];
        const size_t numJB = cell1.boundedBy().size();
        CGP_ASSERT_OP(numJB,<=,2);

        typedef std::set<size_t> SetType;
        typedef typename SetType::const_iterator SetIter;
        SetType adjBounadries;

        // loop over all junctions of a boundarie (most the time 2 junctions sometime 1 and zero (rings,boarder))
        for(size_t j=0;j<numJB;++j){

            const size_t ji = cell1.boundedBy()[j]-1;
            const Cell0 & cell0 = cells0[ji];
            const size_t numBJ = cell0.bounds().size();

            CGP_ASSERT_OP(numBJ,>=,3);
            CGP_ASSERT_OP(numBJ,<=,4);

            // loop over all 3 or 4 boundaries of the junction and add them to adjacency
            for(size_t b=0;b<numBJ;++b){

                const size_t otherBi = cell0.bounds()[b]-1; 
                if(otherBi!=bi){
                    adjBounadries.insert(otherBi);
                }
            }
        }

        // frome here on boundary adj. is known
        const size_t numAdj = adjBounadries.size();
        
        if (numAdj>0){
            // mix the values
            float alphaFrac     = alpha/float(numAdj);
            const float ownVal  = cell1Features(bi);
            float valSum        = 0.0;
            float wSum          = 0.0;

            // add onw value

            valSum += (1.0-alpha)*ownVal;
            wSum   += (1.0-alpha);



            for(SetIter iter = adjBounadries.begin();iter!=adjBounadries.end();++iter){
                const size_t otherBi  = *iter;
                const float  otherVal = cell1Features(otherBi);
                const float  wColor   = std::exp(-1.0*gamma*std::abs(ownVal-otherVal));
                const float  wTotal   = wColor*alphaFrac;

                // add other values
                valSum +=wTotal*otherVal;
                wSum   +=wTotal;
            }
            const float newVal = valSum / wSum;
            resultArray(bi)=newVal;
        }
    }

    return resultArray;
}




template<class CGP>
vigra::NumpyAnyArray  graphPropagation(
    const CGP & cgp,
    vigra::NumpyArray<1,float> cell1Features
){

    typedef vigra::NumpyArray<1,float>  ResultArray;
    typedef typename ResultArray::difference_type ShapeType;

    typedef typename CGP::Cells0 Cells0;
    typedef typename CGP::Cells1 Cells1;

    typedef typename CGP::Cell0 Cell0;
    typedef typename CGP::Cell1 Cell1;

    const size_t numJunctions = cgp.numCells(0);
    const size_t numBoundaries     = cgp.numCells(1);
    const Cells1 & cells1=cgp.geometry1();
    const Cells0 & cells0=cgp.geometry0();

    // initialize result array
    ResultArray resultArray = ResultArray(ShapeType(numBoundaries));
    std::copy(cell1Features.begin(),cell1Features.end(),resultArray.begin());


    float fw[4]={0.0, 0.0, 0.0, 0.0};


    int a[] = { 3, 1, 0, 4 } ;
    std::vector<size_t> indexes ;
    
    std::vector<float> reservedVec;
    reservedVec.reserve(2);
    std::vector<  std::vector<float>  >  weightCollection(numBoundaries,reservedVec);


    for(size_t j=0;j<numJunctions;++j){
        const Cell0 & cell0 = cells0[j];
        const size_t numBounds = cell0.bounds().size();

        CGP_ASSERT_OP(numBounds,<=,4);
        CGP_ASSERT_OP(numBounds,>=,3);

        //std::cout<<"boundssize "<<numBounds<<"\n";
        // get maximum of junction
        for(size_t b=0;b<numBounds;++b){
            const size_t faceIndex=cell0.bounds()[b]-1;
            const float faceWeight= cell1Features(faceIndex);
            fw[b]=faceWeight;
        }
        
        argsort(fw, fw+numBounds, std::greater<float>(), indexes) ;

        if(numBounds==3){
            const size_t i0 = indexes[0];
            const size_t i1 = indexes[1];
            const size_t i2 = indexes[2];

            const float mean2 = ( fw[i0] + fw[i1] )/2.0;
            const float toGiveAway =( fw[i2]  )/2.0;

            weightCollection[ cell0.bounds()[i0]-1].push_back(mean2+toGiveAway/2.0);
            weightCollection[ cell0.bounds()[i1]-1].push_back(mean2+toGiveAway/2.0);
            weightCollection[ cell0.bounds()[i2]-1].push_back(toGiveAway);
        }
        else{
            const size_t i0 = indexes[0];
            const size_t i1 = indexes[1];
            const size_t i2 = indexes[2];
            const size_t i3 = indexes[3];
            
            const float mean2 = ( fw[i0] + fw[i1] )/2.0;
            const float toGiveAway =( fw[i2] +fw[i3]  )/4.0;

            weightCollection[ cell0.bounds()[i0]-1].push_back(mean2+toGiveAway/2.0);
            weightCollection[ cell0.bounds()[i1]-1].push_back(mean2+toGiveAway/2.0);
            weightCollection[ cell0.bounds()[i2]-1].push_back(toGiveAway);
            weightCollection[ cell0.bounds()[i3]-1].push_back(toGiveAway);
        }
    }

    for(size_t i=0;i<numBoundaries;++i){
        const size_t numWeights=weightCollection[i].size();
        if(numWeights==0){
            resultArray(i)=cell1Features(i);
        }
        else{
            float mean=0;
            for(size_t j=0;j<numWeights;++j){
                mean+=weightCollection[i][j];
            }
            resultArray(i)=mean/numWeights;
        }
    }


    return resultArray;
}



/*
    - min 
    - max
    - mean
    - median
*/

template<class CGP>
vigra::NumpyAnyArray  cell1GraphAdjAcc(
    const CGP & cgp,
    vigra::NumpyArray<1,float> cell1Features
){

    typedef vigra::NumpyArray<2,float>  ResultArray;
    typedef typename ResultArray::difference_type ShapeType;
    typedef typename CGP::Cells0 Cells0;
    typedef typename CGP::Cells1 Cells1;
    typedef typename CGP::Cell0 Cell0;
    typedef typename CGP::Cell1 Cell1;
    typedef typename CGP::CellAdjacencyGraphVectorType CellAdjGraph;

    const size_t cellType = 1;
    const size_t numCells = cgp.numCells(cellType);

    using namespace boost::accumulators;
    typedef accumulator_set<double, stats<
        tag::min,
        tag::max,
        tag::mean,
        tag::median(with_p_square_quantile)
    > > AccSet;


     // initialize result array
    ResultArray resultArray = ResultArray(ShapeType(numCells,4));

    // Adj Graph 
    CellAdjGraph cellAdjGraph;
    //std::cout<<"fill adj\n";
    // fill adjacency graph
    cgp.cellAdjacencyGraphVector(cellType,cellAdjGraph);
    //std::cout<<"fill adj done\n";


    for(size_t ci=0;ci<numCells;++ci){

        AccSet accSet;
        accSet(static_cast<double>(cell1Features(ci)));

        for(size_t n=0;n<cellAdjGraph[ci].size();++n){
            accSet(static_cast<double>(cell1Features(cellAdjGraph[ci][n]-1)));
        }

        resultArray(ci,0)=static_cast<float>(boost::accumulators::min(accSet));
        resultArray(ci,1)=static_cast<float>(boost::accumulators::max(accSet));
        resultArray(ci,2)=static_cast<float>(boost::accumulators::mean(accSet));
        resultArray(ci,3)=static_cast<float>(boost::accumulators::median(accSet));
    }
    return resultArray;
}





template<class CGP>
vigra::NumpyAnyArray  graphMax(
    const CGP & cgp,
    vigra::NumpyArray<1,float> cell1Features
){

    typedef vigra::NumpyArray<1,float>  ResultArray;
    typedef typename ResultArray::difference_type ShapeType;

    typedef typename CGP::Cells0 Cells0;
    typedef typename CGP::Cells1 Cells1;

    typedef typename CGP::Cell0 Cell0;
    typedef typename CGP::Cell1 Cell1;

    const size_t numJunctions = cgp.numCells(0);
    const size_t numBoundaries     = cgp.numCells(1);
    const Cells1 & cells1=cgp.geometry1();
    const Cells0 & cells0=cgp.geometry0();

    // initialize result array
    ResultArray resultArray = ResultArray(ShapeType(numBoundaries));
    std::copy(cell1Features.begin(),cell1Features.end(),resultArray.begin());


    for(size_t j=0;j<numJunctions;++j){
        float maxFaceWeight = -1.0*std::numeric_limits<float>::infinity();
        const Cell0 & cell0 = cells0[j];
        const size_t numBounds = cell0.bounds().size();
        //std::cout<<"boundssize "<<numBounds<<"\n";
        // get maximum of junction
        for(size_t b=0;b<numBounds;++b){
            const size_t faceIndex=cell0.bounds()[b]-1;
            const float faceWeight= cell1Features(faceIndex);
            maxFaceWeight = faceWeight>maxFaceWeight ? faceWeight : maxFaceWeight ;
        }
        // get maximum of junction
        for(size_t b=0;b<numBounds;++b){
            const size_t faceIndex=cell0.bounds()[b]-1;
            resultArray(faceIndex)=maxFaceWeight;
        }
    }
    return resultArray;
}



template<class CGP>
vigra::NumpyAnyArray cell1BoundsArray(
    const CGP & cgp
){
    typedef vigra::NumpyArray<2,int>  ResultArray;
    typedef typename ResultArray::difference_type ShapeType;

    typedef typename CGP::Cells0 Cells0;
    typedef typename CGP::Cells1 Cells1;
    typedef typename CGP::Cells2 Cells2;

    typedef typename CGP::Cell0 Cell0;
    typedef typename CGP::Cell1 Cell1;
    typedef typename CGP::Cell2 Cell2;

    const size_t numBoundaries = cgp.numCells(1);
    const size_t numRegion     = cgp.numCells(2);
    const Cells1 & cells1=cgp.geometry1();
    CGP_ASSERT_OP(numBoundaries,==,cells1.size());

    ResultArray resultArray = ResultArray(ShapeType(numBoundaries,2));

    for(size_t b=0;b<numBoundaries;++b){
        const Cell1 & cell1 = cells1[b];
        const int la = cell1.bounds()[0];
        const int lb = cell1.bounds()[1];
        CGP_ASSERT_OP(la,!=,0);
        CGP_ASSERT_OP(lb,!=,0);
        resultArray(b,0)=la;
        resultArray(b,1)=lb;
    }
    return resultArray;
}


template<class CELL>
python::tuple pointNumpyTupe(const CELL & cell){
    const size_t numPoints=cell.points().size();
    typedef vigra::NumpyArray<1,vigra::UInt32>  SingleCoordArrayType;
    typedef typename SingleCoordArrayType::difference_type ShapeType;
    const ShapeType shape(numPoints);

    SingleCoordArrayType cx(shape),cy(shape);
    for(size_t i=0;i<numPoints;++i){
        cx(i)=cell.points()[i][0];
        cy(i)=cell.points()[i][1];
    }
    vigra::NumpyAnyArray ax=cx,ay=cy;
    return python::make_tuple(ax,ay);
}


template<class TGRID>
vigra::NumpyAnyArray getCellLabelGrid(
                                const TGRID & tgrid,
                                int cellType,
                                bool useTopologicalShape,
                                vigra::NumpyArray<2, vigra::Singleband<npy_uint32> > res = vigra::NumpyArray<2,vigra::Singleband<npy_uint32> >()){

    if(useTopologicalShape){
        res.reshapeIfEmpty(tgrid.tgrid().shape());
        std::fill(res.begin(),res.end(),0);

        if(cellType==0){
            for(size_t y=1;y<tgrid.shape(1);y+=2)
            for(size_t x=1;x<tgrid.shape(0);x+=2){
                if( tgrid(x,y)!=0 )
                    res(x,y)=tgrid(x,y);
            }
        }
        else if(cellType==1){
            for(size_t y=0;y<tgrid.shape(1);++y)
            for(size_t x=0;x<tgrid.shape(0);++x){
                if(  (  (x%2==0 && y%2!=0) || (x%2!=0 && y%2==0) ) && tgrid(x,y)!=0 )
                    res(x,y)=tgrid(x,y);
            }
        }
        else if(cellType==2){
            for(size_t y=0;y<tgrid.shape(1);y+=2)
            for(size_t x=0;x<tgrid.shape(0);x+=2){
                res(x,y)=tgrid(x,y);
            }
        }
    }
    else{
        typedef typename TGRID::LabelImageType::difference_type ShapeType;
        const ShapeType shape( (tgrid.shape(0)+1)/2,(tgrid.shape(1)+1)/2   );
        res.reshapeIfEmpty(shape);
        std::fill(res.begin(),res.end(),0);

        if(cellType==0){
            for(size_t y=1;y<tgrid.shape(1);y+=2)
            for(size_t x=1;x<tgrid.shape(0);x+=2){
                if( tgrid(x,y)!=0 ){
                    // for junction-pixel:
                    // 1 pixel in t-grid => 4 pixels in grid 
                    res( (x+1)/2,(y+1)/2 )=tgrid(x,y);
                    res( (x+1)/2,(y-1)/2 )=tgrid(x,y);
                    res( (x-1)/2,(y+1)/2 )=tgrid(x,y);
                    res( (x-1)/2,(y+1)/2 )=tgrid(x,y);
                }
            }
        }

        else if(cellType==1){
            for(size_t y=0;y<tgrid.shape(1);++y)
            for(size_t x=0;x<tgrid.shape(0);++x){

                if(tgrid(x,y)!=0 ){

                    // - boundary 
                    if( x%2==0 && y%2!=0 ){
                        // for boundary-pixel:
                        // 1 pixel in t-grid => 2 pixels in grid 
                        res( x/2,(y+1)/2 )=tgrid(x,y);
                        res( x/2,(y-1)/2 )=tgrid(x,y);
                    }
                    //  |  boundary
                    else if(x%2!=0 && y%2==0){
                        // for boundary-pixel:
                        // 1 pixel in t-grid => 2 pixels in grid 
                        res( (x-1)/2,y/2 )=tgrid(x,y);
                        res( (x+1)/2,y/2 )=tgrid(x,y);
                    }
                }
            }
        }

        else if(cellType==2){
            for(size_t y=0;y<tgrid.shape(1);y+=2)
            for(size_t x=0;x<tgrid.shape(0);x+=2){
                // for region-pixel:
                // 1 pixel in t-grid => 4 pixels in grid 
                res(x/2,y/2)=tgrid(x,y);
            }
        }   
    }
    return res;
}


template<class CGP>
vigra::NumpyArray<1, unsigned int> pyCgpSerialize(
    const CGP& cgp
) {
    vigra::NumpyArray<1, unsigned int> result;
    std::vector<unsigned int> res = cgp.serialize();
    result.reshape(vigra::Shape1(res.size()));
    std::copy(res.begin(), res.end(), result.begin());
    return result;
}

template<class CGP>
const typename CGP::TopologicalGridType * merge2Cells(
    const CGP & cgp,
    vigra::NumpyArray<1,npy_uint32> cell1States
){
    typename CGP::TopologicalGridType * tgrid = new typename CGP::TopologicalGridType();
    cgp.merge2Cells(cell1States.begin(),cell1States.end(),*tgrid);
    return tgrid;
}





template<class CGP>
vigra::NumpyAnyArray featuresToFeatureImage(
    const CGP & cgp,
    int cellType,
    vigra::NumpyArray<1,float> features,
    const bool ignoreInactive,
    const float inactiveValue,
    const bool useTopologicalShape=true,
    vigra::NumpyArray<2, vigra::Singleband<float> > res = vigra::NumpyArray<2,vigra::Singleband<float> >()
){
    typedef typename  CGP::LabelType LabelType;
    if(useTopologicalShape)
        res.reshapeIfEmpty(cgp.tgrid().tgrid().shape());
    else
        res.reshapeIfEmpty(cgp.tgrid().shapeLabeling());
        

    if(ignoreInactive==false)
        std::fill(res.begin(),res.end(),inactiveValue);

    if(useTopologicalShape){
        if (cellType==0){
            for(size_t y=0;y<cgp.shape(1);++y)
            for(size_t x=0;x<cgp.shape(0);++x){
                const LabelType cellLabel=cgp(x,y);
                if(cellType==0)
                if( (x%2!=0 && y%2!=0) && cellLabel!=0)
                    res(x,y)=features[cellLabel-1];
            }
        }
        if(cellType==1){
            for(size_t y=0;y<cgp.shape(1);++y)
            for(size_t x=0;x<cgp.shape(0);++x){
                const LabelType cellLabel=cgp(x,y);
                if( ( (x%2==0 && y%2!=0)  || (x%2!=0 && y%2==0 ) ) && cellLabel!=0)
                    res(x,y)=features[cellLabel-1];
            }
        }
        else if(cellType==2){
            for(size_t y=0;y<cgp.shape(1);++y)
            for(size_t x=0;x<cgp.shape(0);++x){
                const LabelType cellLabel=cgp(x,y);
                if( (x%2==0 && y%2==0))
                    res(x,y)=features[cellLabel-1];
            }
        }
    }


    else{

        if(cellType==0){
            for(size_t y=1;y<cgp.shape(1);y+=2)
            for(size_t x=1;x<cgp.shape(0);x+=2){
                if( cgp(x,y)!=0 ){
                    // for junction-pixel:
                    // 1 pixel in t-grid => 4 pixels in grid 
                    res( (x+1)/2,(y+1)/2 )=features[cgp(x,y)-1];
                    res( (x+1)/2,(y-1)/2 )=features[cgp(x,y)-1];
                    res( (x-1)/2,(y+1)/2 )=features[cgp(x,y)-1];
                    res( (x-1)/2,(y+1)/2 )=features[cgp(x,y)-1];
                }
            }
        }

        else if(cellType==1){
            for(size_t y=0;y<cgp.shape(1);++y)
            for(size_t x=0;x<cgp.shape(0);++x){

                if(cgp(x,y)!=0 ){

                    // - boundary 
                    if( x%2==0 && y%2!=0 ){
                        // for boundary-pixel:
                        // 1 pixel in t-grid => 2 pixels in grid 
                        res( x/2,(y+1)/2 )=features[cgp(x,y)-1];
                        res( x/2,(y-1)/2 )=features[cgp(x,y)-1];
                    }
                    //  |  boundary
                    else if(x%2!=0 && y%2==0){
                        // for boundary-pixel:
                        // 1 pixel in t-grid => 2 pixels in grid 
                        res( (x-1)/2,y/2 )=features[cgp(x,y)-1];
                        res( (x+1)/2,y/2 )=features[cgp(x,y)-1];
                    }
                }
            }
        }

        else if(cellType==2){
            for(size_t y=0;y<cgp.shape(1);y+=2)
            for(size_t x=0;x<cgp.shape(0);x+=2){
                // for region-pixel:
                // 1 pixel in t-grid => 4 pixels in grid 
                res(x/2,y/2)=features[cgp(x,y)-1];
            }
        }   
    }
    return res;
}


template<class CGP>
vigra::NumpyAnyArray orientedWatershedTransform
(   
    const CGP & cgp,
    vigra::NumpyArray<2 ,vigra::TinyVector< float  ,2 > > gradientImage
){
    typedef vigra::NumpyArray<1, float  > NumpyFloat1d;
    typedef typename NumpyFloat1d::difference_type ShapeType;
    typedef vigra::TinyVector<float,2> VecType;

    const size_t numBoundaries = cgp.numCells(1);
    NumpyFloat1d resultWeights=NumpyFloat1d(ShapeType(numBoundaries));

    
    typedef typename CGP::Cells1 Cells1;
    typedef typename CGP::Cell1   Cell1;

    const Cells1 & cells1=cgp.geometry1();

    for(size_t i=0;i<numBoundaries;++i){
        const Cell1 & cell = cells1[i];

        float weight=0.0;
        for (size_t j =0; j<cell.size(); ++j){


            // get "orientation vector" of 
            // boundary and transform it into 
            // normal vector
            VecType normalFace = cell.angles_[j];

            const float gx =  normalFace[0];
            const float gy =  normalFace[1];

            normalFace/=vigra::norm(normalFace);

            float np= std::sqrt(normalFace[0]*normalFace[0] +normalFace[1]*normalFace[1]);
            std::cout<<"np "<<np<<"\n";



            //normalFace[0] = -1.0 *gy;
            //normalFace[1] = gx;


            const size_t x=cell[j][0];
            const size_t y=cell[j][1];
            //'std::cout << "x,y "<<x<<","<<y<<"\n";
            VecType grad = gradientImage(x,y);
            grad/=vigra::norm(grad);

            if(grad[1]>0){
                grad*=-1.0;
            }

            if(normalFace[1]>0){
                normalFace*=-1.0;
            } 



            const float dotP = vigra::dot(grad,normalFace);
            float theta= std::acos(dotP);

            std::cout<<"theta "<< theta<<" "<<"dotP "<<dotP<<" gX,gY "<<grad[0]<<","<<grad[1]<<" nX,nY "<<normalFace[0]<<","<<normalFace[1]<<" \n";
            
            while(theta>=M_PI/2.0){
                theta-=M_PI/2.0;
            }
            theta/=(M_PI/2.0);

            std::cout<<"final theta "<<theta<<"\n";

            if ( std::isnan(theta)){
                theta=0.5;
            }

            weight += theta;
            
        }
        weight/=float(cell.size());
        std::cout<<"********************************final weight "<<weight<<"\n";
        resultWeights(i)=weight;
    }
    
    
    return resultWeights;
}

template<class CGP>
vigra::NumpyAnyArray cellSizes(
    const CGP & cgp,
    const size_t cellType,
    vigra::NumpyArray<1, vigra::UInt32 > res = vigra::NumpyArray<1,vigra::UInt32> ()
){
    const size_t nCells = cgp.numCells(cellType);
    res.reshapeIfEmpty(typename vigra::NumpyArray<1,vigra::UInt32>::difference_type(nCells));
    for(size_t  c=0;c<nCells;++c){
        res(c)=cgp.cellSize(cellType,c);
    }
    return res;
}

void export_cgp2d()
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

    typedef  vigra::NumpyArray<2 ,vigra::Singleband < LabelType > > InputLabelImageType;
    // cgp type and cell types
    typedef CgpType::PointType PointType;
    // bound vector
    typedef std::vector<float> FloatVectorType;
    typedef std::vector<LabelType> LabelVectorType;
    // point vector
    typedef std::vector<PointType> PointVectorType;
    // geo cells 
    typedef CgpType::Cell0 Cell0Type;
    typedef CgpType::Cell1 Cell1Type;
    typedef CgpType::Cell2 Cell2Type;

    typedef CgpType::Cells0 Cell0VectorType;
    typedef CgpType::Cells1 Cell1VectorType;
    typedef CgpType::Cells2 Cell2VectorType;

    // cell vectors
    python::class_<TopologicalGridType>("TopologicalGrid",python::init<const InputLabelImageType & >())
    .add_property("shape", python::make_function(&TopologicalGridType::shapeTopologicalGrid, python::return_value_policy<return_by_value>()) )
    .add_property("shapeLabeling", python::make_function(&TopologicalGridType::shapeLabeling, python::return_value_policy<return_by_value>()) )
    .def("numCells",&TopologicalGridType::numCells)
    .def("labelGrid",vigra::registerConverters(&getCellLabelGrid<TopologicalGridType> ) ,
        (
            arg("cellType"),
            arg("useTopologicalShape")=true,
            arg("out")=python::object() 
        )  
    )
    ;

    // float vector
    python::class_<FloatVectorType>("FloatVector",init<>())
        .def(vector_indexing_suite<FloatVectorType ,true >())
    ;
    // bound / bounded by vector
    python::class_<LabelVectorType> exporter = python::class_<LabelVectorType>("LabelVector",init<>())
        .def(vector_indexing_suite<LabelVectorType >())
    ;
    // point   vector
    python::class_<PointVectorType>("PointVector",init<>())
        .def(vector_indexing_suite<PointVectorType ,true>())
    ;

    /*
    // cells
    python::class_<Cell0Type>("Cell0",python::init<>())
        .def(CellTypeSuite<Cell0Type>())
        .def("getAngles",&getAngles<Cell0Type>,python::return_value_policy<python::manage_new_object>())
    ;

    python::class_<Cell1Type>("Cell1",python::init<>())
        .def(CellTypeSuite<Cell1Type>())
    ;

    python::class_<Cell2Type>("Cell2",python::init<>())
        .def(CellTypeSuite<Cell2Type>())
    ;

    // cells vectors
    python::class_<Cell0VectorType>("Cell0Vector",init<>())
        .def(vector_indexing_suite<Cell0VectorType >())
    ;
    python::class_<Cell1VectorType>("Cell1Vector",init<>())
        .def(vector_indexing_suite<Cell1VectorType >())
    ;
    python::class_<Cell2VectorType>("Cell2Vector",init<>())
        .def(vector_indexing_suite<Cell2VectorType >())
    ;
    */

    /************************************************************************/
    /* C e l l B a s e                                                      */
    /************************************************************************/

    python::class_<CgpType>("Cgp",python::init<const TopologicalGridType & >()[with_custodian_and_ward<1 /*custodian == self*/, 2 /*ward == const TopologicalGridType& */>()] )

        .add_property("shape", python::make_function(&CgpType::shapeTopologicalGrid, python::return_value_policy<return_by_value>()) )
        .add_property("shapeLabeling", python::make_function(&CgpType::shapeLabeling, python::return_value_policy<return_by_value>()))

        
        //.def("shape",&CgpType::shape)
        
        .add_property("tgrid", python::make_function(&CgpType::tgrid, return_internal_reference<>() ))
        .add_property("cells0", python::make_function(&CgpType::geometry0, return_internal_reference<>() ))
        .add_property("cells1", python::make_function(&CgpType::geometry1, return_internal_reference<>() ))
        .add_property("cells2", python::make_function(&CgpType::geometry2, return_internal_reference<>() ))
        .def("cell1BoundsArray",vigra::registerConverters(&cell1BoundsArray<CgpType>))

        /*
        .def("_cell1countMultiEdges",vigra::registerConverters(&countMultiEdges<CgpType>))
        .def("_cell1TopoFeatures",vigra::registerConverters(&cell1TopoFeatures<CgpType>))
        .def("_cell1GeoFeatures",vigra::registerConverters(&cell1GeoFeatures<CgpType>))
        .def("_cell1RelativeCenterDist",vigra::registerConverters(&relativeCenterDist<CgpType>))
        .def("_cell1BoarderTouch",vigra::registerConverters(&boarderTouch<CgpType>))
        .def("_cell1GraphStealing",vigra::registerConverters(&cell1GraphStealing<CgpType>))
        .def("_cell1GraphMax", vigra::registerConverters(&graphMax<CgpType>))
        .def("_cell1GraphPropagation",vigra::registerConverters(&graphPropagation<CgpType>))
        .def("_cell1GraphBiMean",vigra::registerConverters(&graphBiMean<CgpType>))
        .def("_cell1GraphAdjAcc",vigra::registerConverters(&cell1GraphAdjAcc<CgpType>))
        */
        .def("cellSizes",vigra::registerConverters(&cellSizes<CgpType>),
            (
                arg("cellType"),
                arg("out")=python::object()
            )
        )
        .def("serialize", &pyCgpSerialize<CgpType>)
        .def("numCells",&CgpType::numCells)
        .def("featureToImage",vigra::registerConverters(&featuresToFeatureImage<CgpType>),
            (
                arg("cellType"),
                arg("features"),
                arg("ignoreInactive")=false,
                arg("inactiveValue")=0.0f,
                arg("useTopologicalShape")=true,
                arg("out")=python::object()
            )
        )
        .def("merge2Cells",vigra::registerConverters( &merge2Cells<CgpType> ) ,python::return_value_policy<python::manage_new_object>(),
            (
                arg("cell1States")
            )
        )
        .def("owt", vigra::registerConverters(&orientedWatershedTransform<CgpType> ))
    ;
}

} // namespace vigra

