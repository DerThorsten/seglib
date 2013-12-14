    
#ifndef CGP2D_DYNAMIC_GRAPH_HXX
#define CGP2D_DYNAMIC_GRAPH_HXX

/* std library */
#include <set>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <deque>
#include <map>
#include <stdexcept>
#include <sstream>

/* vigra */
#include <vigra/multi_array.hxx>
#include <vigra/tinyvector.hxx>
#include <vigra/multi_array.hxx>

/* opengm */
//#include "opengm/config.hxx"
//#include "opengm/utilities/metaprogramming.hxx"

/* this project */
#include "partition.hxx"
#include "macros.hxx"

#include "utility/line.hxx"
#include "cells/cell_base.hxx"
#include "cells/cell0.hxx"
#include "cells/cell1.hxx"
#include "cells/cell2.hxx"
#include "tgrid.hxx"
#include "cgp2d.hxx"


namespace cgp2d {

//class DynamicGraph;
class EdgeMapBase;
class NodeMapBase;


struct DgraphNode{
    std::set<size_t> edges_;
};

class DynamicGraph{

    friend class EdgeMapBase;  
    friend class NodeMapBase;
public:
    typedef std::pair< size_t,size_t>   EdgeType;
    typedef DgraphNode                  NodeType;


    // setup
    DynamicGraph(const size_t nNodes,const size_t nEdges);
    void   setInitalEdge(const size_t initEdge,const size_t initNode0,const size_t initNode1);

    // query
    size_t numberOfNodes()const;
    size_t numberOfEdges()const;
    size_t initNumberOfNodes()const;
    size_t initNumberOfEdges()const;


    size_t getAndEdge()const{
        return dynamicEdges_.begin()->first;
    }

    template<class OUT_ITER>
    void activeNodeLabels(OUT_ITER begin,OUT_ITER end)const;
    template<class OUT_ITER>
    void activeEdgeLabels(OUT_ITER begin,OUT_ITER end)const;


    // modification
    void mergeParallelEdges();

    void mergeRegions(const size_t edgeIndex);

private:


    typedef std::map<size_t, EdgeType > EdgeMap;
    typedef std::map<size_t, NodeType > NodeMap;
    typedef typename EdgeMap::const_iterator ConstEdgeMapIterator;
    typedef typename NodeMap::const_iterator ConstNodeMapIterator;


    void combineDoubleEdges(const std::vector<size_t> & ,const size_t ,const size_t );

    
    void registerMap( NodeMapBase * mapPtr){
        nodeMaps_.push_back(mapPtr);
    }
    void registerMap(EdgeMapBase * mapPtr){
        edgeMaps_.push_back(mapPtr);
    }

    size_t nInitNodes_;
    size_t nInitEdges_;

    partition::Partition<size_t> nodeUfd_;
    partition::Partition<size_t> edgeUfd_;

    std::vector< EdgeType >     initEdges_;
    

    EdgeMap dynamicEdges_;
    NodeMap dynamicNodes_;


    std::vector< NodeMapBase * > nodeMaps_;
    std::vector< EdgeMapBase * > edgeMaps_;

};

class NodeMapBase{
public:
    NodeMapBase() {
    }
    virtual ~NodeMapBase(){}
    virtual void registerMap(DynamicGraph & dgraph) {
        dgraph.registerMap(this);
    }
    virtual void merge(const std::vector<size_t> & toMerge,const size_t newIndex)=0;
};


class EdgeMapBase{
public:
    EdgeMapBase() {
    }
    virtual ~EdgeMapBase(){}
    void registerMap(DynamicGraph & dgraph) {
        dgraph.registerMap(this);
    }
    virtual void merge(const std::vector<size_t> & toMerge,const size_t newIndex)=0;
    virtual void erase(const size_t index)=0;
};




template<class T>
class EdgeFeatureMap : public EdgeMapBase {
public:
    typedef T value_type;
    // dummy
    EdgeFeatureMap(){}
    EdgeFeatureMap(
        DynamicGraph & dgraph ,
        const vigra::MultiArrayView<2,T>        & features,
        const vigra::MultiArrayView<1,size_t>   & edgeSize
    ) 
    :   EdgeMapBase(),
        features_(features),
        featuresBuffer_(typename vigra::MultiArray<1,T>::difference_type(features.shape(1))),
        edgeSize_(edgeSize)
    {   
        this->registerMap(dgraph);
        CGP_ASSERT_OP(features.shape(0),==,edgeSize.shape(0));
        CGP_ASSERT_OP(features.shape(0),==,dgraph.initNumberOfEdges());
    }
    virtual void merge(const std::vector<size_t> & toMerge,const size_t newIndex){
        featuresBuffer_=static_cast<T>(0.0);
        size_t newSize=0;

        for(size_t tm=0;tm<toMerge.size();++tm){
            // get edge index and size of to merge edge
            const size_t tmi=toMerge[tm];
            const size_t size=edgeSize_[tmi];
            CGP_ASSERT_OP(size,>,0);
            // set the size to zero (newIndexes size will be set !=0 later)
            edgeSize_[tmi]=0;
            
            // accumulate new size
            newSize+=size;

            // accumulate features
            for(size_t fi=0;fi<numberOfFeatures();++fi){
                featuresBuffer_(fi)+=static_cast<T>(size)*features_(tmi,fi);
            }
        }
        // normalize and write to new Index
        edgeSize_[newIndex]=newSize;
        for(size_t fi=0;fi<numberOfFeatures();++fi){
            features_(newIndex,fi)=featuresBuffer_(fi)/static_cast<T>(newSize);
        }

    }
    virtual void erase(const size_t index){
        // set size to zero just for assertions
        edgeSize_[index]=0;
    }

    size_t numberOfFeatures()const{
        return features_.shape(1);
    }



private:
    vigra::MultiArrayView<2,T>       features_;
    vigra::MultiArray<1,T>       featuresBuffer_;
    vigra::MultiArrayView<1,size_t>  edgeSize_;
};





DynamicGraph::DynamicGraph(const size_t nNodes,const size_t nEdges)
:   nInitNodes_(nNodes),
    nInitEdges_(nEdges),
    nodeUfd_(nNodes),
    edgeUfd_(nEdges),
    initEdges_(nEdges)
{
     // fill nodes
    for(size_t n=0;n<nNodes;++n){
        dynamicNodes_[n]=NodeType();
        CGP_ASSERT_OP(nodeUfd_.find(n),==,n);
    }
}

void DynamicGraph::setInitalEdge(const size_t initEdge,const size_t initNode0,const size_t initNode1){

    // set up inital and dynamic edges
    initEdges_[initEdge].first =initNode0;
    initEdges_[initEdge].second=initNode1;
    dynamicEdges_[initEdge] = initEdges_[initEdge];

    // set up the edges of a given region mapping
    dynamicNodes_[initNode0].edges_.insert(initEdge);
    dynamicNodes_[initNode1].edges_.insert(initEdge);
}

inline size_t DynamicGraph::numberOfNodes()const{
    return dynamicNodes_.size();
}

inline size_t DynamicGraph::numberOfEdges()const{
    return dynamicEdges_.size();
}

inline size_t DynamicGraph::initNumberOfNodes()const{
    return nInitNodes_;
}

inline size_t DynamicGraph::initNumberOfEdges()const{
    return nInitEdges_;
}

template<class OUT_ITER>
void DynamicGraph::activeNodeLabels(OUT_ITER begin,OUT_ITER end)const{
    CGP_ASSERT_OP(std::distance(begin,end),==,this->numberOfNodes());
}
template<class OUT_ITER>
void DynamicGraph::activeEdgeLabels(OUT_ITER begin,OUT_ITER end)const{
    CGP_ASSERT_OP(std::distance(begin,end),==,this->numberOfEdges());

    //for(ConstEdgeMapIterator iter=dynamicEdges_.begin();)
}

void DynamicGraph::mergeParallelEdges(){
    typedef std::map< size_t , std::vector<size_t> > DoubleEdgeMap;
    typedef typename DoubleEdgeMap::const_iterator MapIter;
    DoubleEdgeMap pEdgeFinder;
    for(size_t e=0;e<nInitEdges_;++e){
        size_t n0=initEdges_[e].first;
        size_t n1=initEdges_[e].second;
        if(n0<n1){
            std::swap(n0,n1);
        }
        const size_t key = n0 + nInitNodes_*n1;
        pEdgeFinder[key].push_back(e);
    }

    for(MapIter iter=pEdgeFinder.begin();iter!=pEdgeFinder.end();++iter){
        const std::vector<size_t> & dEdges = iter->second;
        CGP_ASSERT_OP(dEdges.size(),!=,0);

        if(dEdges.size()>1){
            //std::cout<<"found double edges "<<dEdges.size()<<"\n";
            const size_t key = iter->first;
            const size_t r1  = key/nInitNodes_;
            const size_t r0  = key - nInitNodes_*r1;
            this->combineDoubleEdges(dEdges,r0,r1);
        }
    }
}

void DynamicGraph::combineDoubleEdges(const std::vector<size_t> & toCombine,const size_t r0,const size_t r1){
    std::set<size_t> toCombineSet(toCombine.begin(),toCombine.end());
    CGP_ASSERT_OP(toCombine.size(),==,toCombineSet.size());



    CGP_ASSERT_OP(dynamicEdges_.size(),==,edgeUfd_.numberOfSets());
    // merge in ufd
    const size_t firstElement=toCombine.front();
    for(size_t i=1;i<toCombine.size();++i){
        edgeUfd_.merge(toCombine[i],firstElement);
    }
    // new representative index
    const size_t newIndex=edgeUfd_.find(firstElement);

    // delete |toCombine|-1 edges in dynamic map
    for(size_t i=0;i<toCombine.size();++i){
        if(toCombine[i]!=newIndex){
            const bool found = static_cast<bool>(dynamicEdges_.find(toCombine[i])!=dynamicEdges_.end());
            CGP_ASSERT_OP(found,==,true);
            dynamicEdges_.erase(toCombine[i]);
        }
    }

    // call registerMaped edge maps merge
    for(size_t m=0; 
        m<edgeMaps_.size();++m){
        edgeMaps_[m]->merge(toCombine,newIndex);
    }

    // update the two region between the double edge 
    const size_t regions[2]={r0,r1};
    for(size_t r=0;r<2;++r){
        const size_t ri=regions[r];
        std::set<size_t> & nodesEdges = dynamicNodes_[ri].edges_;
        for(size_t i=0;i<toCombine.size();++i){

            if(toCombine[i]!=newIndex){
                const bool found = static_cast<bool>(nodesEdges.find(toCombine[i])!=nodesEdges.end());
                CGP_ASSERT_OP(found,==,true);
                const size_t nErased = nodesEdges.erase(toCombine[i]);
                CGP_ASSERT_OP(nErased,==,1);
            }
        }
    }

    CGP_ASSERT_OP(dynamicEdges_.size(),==,edgeUfd_.numberOfSets());
}


void DynamicGraph::mergeRegions(const size_t edgeIndex){
    std::cout<<"merge edge "<<edgeIndex<<"\n";
    const size_t preNumNodes = this->numberOfNodes();


    // assertions that edge is active and
    // its own repr.
    CGP_ASSERT_OP(edgeUfd_.find(edgeIndex),==,edgeIndex);
    const bool foundEdge =  dynamicEdges_.find(edgeIndex)!=dynamicEdges_.end();
    CGP_ASSERT_OP(foundEdge,==,true);


    const size_t nodes[2]= {dynamicEdges_[edgeIndex].first,dynamicEdges_[edgeIndex].second };
    CGP_ASSERT_OP(nodes[0],!=,nodes[1]);


    for(size_t n=0;n<2;++n){
        // assertions that node is active and
        // its own repr.
        const size_t  ni=nodes[n];
        CGP_ASSERT_OP(nodeUfd_.find(ni),==,ni);
        const bool foundNode =  dynamicNodes_.find(ni)!=dynamicNodes_.end();
        CGP_ASSERT_OP(foundNode,==,true);
    }



    // merge the two nodes
    nodeUfd_.merge(nodes[0],nodes[1]);
    const size_t newNodeRep    = nodeUfd_.find(nodes[0]);
    const size_t notNewNodeRep =  (newNodeRep == nodes[0] ? nodes[1] : nodes[0] );

    const size_t  edgeSizeRep    = dynamicNodes_[newNodeRep].edges_.size();
    const size_t  edgeSizeNotRep = dynamicNodes_[notNewNodeRep].edges_.size();

    // merge the edges of the nodes
    dynamicNodes_[newNodeRep].edges_.insert(
        dynamicNodes_[notNewNodeRep].edges_.begin(),
        dynamicNodes_[notNewNodeRep].edges_.end()
    );
    std::cout<<"edgeSizeRep "<<edgeSizeRep<<" edgeSizeNotRep "<<edgeSizeNotRep<<"\n";
    CGP_ASSERT_OP(dynamicNodes_[newNodeRep].edges_.size(),==,edgeSizeRep+edgeSizeNotRep-1);

    // delete the old region
    dynamicNodes_.erase(notNewNodeRep);

    // delete the "toDelteEdge "
    dynamicNodes_[newNodeRep].edges_.erase(edgeIndex);
    dynamicEdges_.erase(edgeIndex);
    CGP_ASSERT_OP(dynamicNodes_[newNodeRep].edges_.size(),==,edgeSizeRep+edgeSizeNotRep-2);


    // get all edges of the two nodes
    //  - find the edge between n0 and n1 (and delte it)
    //  - all other edges might be considered for merged
    //    between each other

    std::map<size_t , std::vector<size_t> > doubleEdgeMap_;

    // loop over all edges of the new formed region
    for(
        std::set<size_t>::const_iterator  edgeIter = dynamicNodes_[newNodeRep].edges_.begin();
        edgeIter!=dynamicNodes_[newNodeRep].edges_.end();
        ++edgeIter
    ){
        const size_t outEdgeIndex = *edgeIter;
        CGP_ASSERT_OP(outEdgeIndex,!=,edgeIndex);

        const size_t oldNodes[2]= {dynamicEdges_[outEdgeIndex].first,dynamicEdges_[outEdgeIndex].second };
        // do the relabling 
        size_t newNodes[2]={
            oldNodes[0]==notNewNodeRep ? newNodeRep : oldNodes[0] , 
            oldNodes[1]==notNewNodeRep ? newNodeRep : oldNodes[1]
        };
        if(newNodes[1]<newNodes[0]){
            std::swap(newNodes[1],newNodes[0]);
        }
        const size_t  key = newNodes[0] + newNodes[1]*this->initNumberOfNodes();
        doubleEdgeMap_[key].push_back(outEdgeIndex);

        // make regions of the edges tidy (even if they might me merged later)

        if(oldNodes[0]==notNewNodeRep ){
            dynamicEdges_[outEdgeIndex].first = newNodeRep;
        }
        else if(oldNodes[1]==notNewNodeRep ){
            dynamicEdges_[outEdgeIndex].second = newNodeRep;
        }
        if(oldNodes[0]==notNewNodeRep && oldNodes[1]==notNewNodeRep){
            CGP_ASSERT_OP(true,==,false);
        }

    }

    // loop over the double map an
    for(
        std::map<size_t , std::vector<size_t> >::const_iterator dIter = doubleEdgeMap_.begin();
        dIter!=doubleEdgeMap_.end();
        ++dIter
    ){
        const std::vector<size_t> edgeVec = dIter->second;
        if(edgeVec.size()>=2){

            std::cout<<"edges to merge\n";
            const size_t ftmi = edgeVec.front();

            for(size_t tm=1;tm<edgeVec.size();++tm){
                const size_t tmi = edgeVec[tm];
                edgeUfd_.merge(ftmi,tmi);
            }
            const size_t newEdgeRep = edgeUfd_.find(ftmi);




            // delte all edges but newEdgeRep
            for(size_t tm=0;tm<edgeVec.size();++tm){
                const size_t tmi = edgeVec[tm];
                if(tmi!=newEdgeRep){

                    dynamicNodes_[newNodeRep].edges_.erase(tmi);
                    const bool foundTmiEdge =  dynamicEdges_.find(tmi)!=dynamicEdges_.end();
                    CGP_ASSERT_OP(foundTmiEdge,==,true);


                    // change the other regions 
                    const size_t otherRis[2]={dynamicEdges_[tmi].first,dynamicEdges_[tmi].second};
                    CGP_ASSERT_OP(otherRis[0],!=,otherRis[1]);
                    size_t toUpdateRi=0;
                    if(otherRis[0]==newNodeRep){ toUpdateRi=otherRis[1];}
                    else if(otherRis[1]==newNodeRep){ toUpdateRi=otherRis[0];}
                    else{CGP_ASSERT_OP(false,==,true);}

                    // get the edges of the "other region" to update
                    std::set<size_t> & toUpdateEdges = dynamicNodes_[toUpdateRi].edges_;
                    const size_t nErased = toUpdateEdges.erase(tmi);
                    CGP_ASSERT_OP(nErased,==,1);
                    const bool inserted = toUpdateEdges.insert(newEdgeRep).second;
                    //CGP_ASSERT_OP(inserted,==,true);


                    dynamicEdges_.erase(tmi);

                }
            }

            // call merge for edge maps
            for(size_t m=0;m<edgeMaps_.size();++m){
                edgeMaps_[m]->merge(edgeVec,newEdgeRep);
                edgeMaps_[m]->erase(edgeIndex);
            }
        } 
        // call merge for node maps
        std::vector<size_t> toMergeNodes(&nodes[0],&nodes[2]);
        for(size_t m=0;m<nodeMaps_.size();++m){
            nodeMaps_[m]->merge(toMergeNodes,newNodeRep);
        }
    }




    CGP_ASSERT_OP(dynamicNodes_.size(),==,preNumNodes-1);
    CGP_ASSERT_OP(nodeUfd_.numberOfSets(),==,preNumNodes-1);
    CGP_ASSERT_OP(this->numberOfNodes(),==,preNumNodes-1);
}




}



#endif //CGP2D_DYNAMIC_GRAPH_HXX