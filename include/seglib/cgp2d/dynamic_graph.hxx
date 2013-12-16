    
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


class DgraphNode{

public:
    typedef std::set<size_t>::const_iterator EdgeIterator;

    // query
    size_t numberOfEdges()const{return edges_.size();}

    // modification
    void  mergeEdges(const DgraphNode & other){
        edges_.insert(other.edges_.begin(),other.edges_.end());
    }

    bool eraseEdge(const size_t edgeIndex){
        return edges_.erase(edgeIndex)==1;
    }

    void eraseAndInsert(const size_t removeEdge,const size_t insertEdge){
        edges_.erase(removeEdge);
        edges_.insert(insertEdge);
    }

    EdgeIterator edgesBegin()const{
        return edges_.begin();
    }
    EdgeIterator edgesEnd()const{
        return edges_.end();
    }


//private:    
    std::set<size_t> edges_;

};


class DgraphEdge{

public:
    bool hasNode(const size_t node)const{
        return node==first || node==second;
    }
    size_t otherNode(const size_t node)const{
        CGP_ASSERT_OP(hasNode(node),==,true);
        return (node==first ? second : first);
    }
    const size_t & operator[](const size_t i)const{
        return (i==0 ? first : second);
    }
    size_t & operator[](const size_t i){
        return (i==0 ? first : second);
    }
//private:
    size_t first;
    size_t second;

};

class DynamicGraph{

    friend class EdgeMapBase;  
    friend class NodeMapBase;
public:
    typedef DgraphEdge   EdgeType;
    typedef DgraphNode   NodeType;


    typedef std::map<size_t , std::vector<size_t>  > DoubleMap;
    // setup
    DynamicGraph(){}
    DynamicGraph(const size_t nNodes,const size_t nEdges);
    void   setInitalEdge(const size_t initEdge,const size_t initNode0,const size_t initNode1);

    // query
    size_t numberOfNodes()const;
    size_t numberOfEdges()const;
    size_t initNumberOfNodes()const;
    size_t initNumberOfEdges()const;


    const EdgeType & getInitalEdge(const size_t index){
        return initEdges_[index];
    }

    const EdgeType & getEdge(const size_t index){
        return dynamicEdges_[index];
    }

    // _NEW means better implementation
    EdgeType getEdge_NEW(const size_t index)const{
        EdgeType edge = initEdges_[index];
        edge[0]=reprNode(edge[0]);
        edge[1]=reprNode(edge[1]);
    }




    const NodeType & getNode(const size_t index){
        return dynamicNodes_[index];
    }

    bool hasEdge(const size_t edgeIndex)const{
        const bool hasEdge  = dynamicEdges_.find(edgeIndex)!=dynamicEdges_.end();
        return hasEdge;
    }

    bool hasEdge_NEW(const size_t edgeIndex)const{
        const EdgeType edge=getEdge_NEW(edgeIndex);
        return edge[0]!=edge[1];
    }

    bool hasNode(const size_t nodeIndex)const{
        const bool hasNode  = dynamicNodes_.find(nodeIndex)!=dynamicNodes_.end();
        return hasNode;
    }

    size_t reprEdge(const size_t edgeIndex)const{
        return edgeUfd_.find(edgeIndex);
    }
    size_t reprNode(const size_t nodeIndex)const{
        return nodeUfd_.find(nodeIndex);
    }


    size_t getAndEdge()const{
        return dynamicEdges_.begin()->first;
    }

    template<class OUT_ITER>
    void activeNodeLabels(OUT_ITER begin,OUT_ITER end)const;
    template<class OUT_ITER>
    void activeEdgeLabels(OUT_ITER begin,OUT_ITER end)const;


    template<class OUT_ITER>
    void stateOfInitalEdges(OUT_ITER begin,OUT_ITER end)const{
        const size_t d = std::distance(begin,end);
        for(size_t ie=0;ie<initNumberOfEdges();++ie){
            const size_t rep=edgeUfd_.find(ie);
            if(dynamicEdges_.find(rep)!=dynamicEdges_.end()){
                begin[ie]=1;
            }
            else{
                begin[ie]=0;
            }
        }
    }


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


    void fillDoubleMapAndRelabelNodes(const NodeType & node , DoubleMap & doubleMap,const size_t relabelFrom,const size_t relabelTo);

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
    virtual void erase(const size_t index,const size_t newNodeIndex)=0;
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

    for(ConstEdgeMapIterator iter=dynamicEdges_.begin();iter!=dynamicEdges_.end();++iter){
        *begin=iter->first;
        ++begin;
    }
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


void DynamicGraph::fillDoubleMapAndRelabelNodes(
    const DynamicGraph::NodeType & node , 
    DynamicGraph::DoubleMap & doubleMap,
    const size_t relabelFrom,
    const size_t relabelTo
){
    // loop over all edges of the new formed region
    for(
        std::set<size_t>::const_iterator  edgeIter = node.edges_.begin();
        edgeIter!=node.edges_.end();
        ++edgeIter
    ){
        const size_t outEdgeIndex = *edgeIter;
        //CGP_ASSERT_OP(outEdgeIndex,!=,edgeIndex);

        const size_t oldNodes[2]= {dynamicEdges_[outEdgeIndex].first,dynamicEdges_[outEdgeIndex].second };
        // do the relabling 
        size_t newNodes[2]={
            oldNodes[0]==relabelFrom ? relabelTo : oldNodes[0] , 
            oldNodes[1]==relabelFrom ? relabelTo : oldNodes[1]
        };
        if(newNodes[1]<newNodes[0]){
            std::swap(newNodes[1],newNodes[0]);
        }
        const size_t  key = newNodes[0] + newNodes[1]*this->initNumberOfNodes();
        doubleMap[key].push_back(outEdgeIndex);

        // make regions of the edges tidy (even if they might me merged later)

        if(oldNodes[0]==relabelFrom ){
            dynamicEdges_[outEdgeIndex].first = relabelTo;
        }
        else if(oldNodes[1]==relabelFrom ){
            dynamicEdges_[outEdgeIndex].second = relabelTo;
        }
        if(oldNodes[0]==relabelFrom && oldNodes[1]==relabelFrom){
            CGP_ASSERT_OP(true,==,false);
        }

    }
}

void DynamicGraph::mergeRegions(const size_t toDeleteEdgeIndex){
    //std::cout<<"merge edge "<<toDeleteEdgeIndex<<"\n";
    const size_t preNumNodes = this->numberOfNodes();

    // assertions that edge is active and
    // its own repr.
    CGP_ASSERT_OP(reprEdge(toDeleteEdgeIndex),==,toDeleteEdgeIndex);
    CGP_ASSERT_OP(hasEdge(toDeleteEdgeIndex),==,true);


    //const size_t nodes[2]= {dynamicEdges_[toDeleteEdgeIndex].first,dynamicEdges_[toDeleteEdgeIndex].second };
    std::vector<size_t> nodes(2);
    nodes[0]=dynamicEdges_[toDeleteEdgeIndex].first;
    nodes[1]=dynamicEdges_[toDeleteEdgeIndex].second;
    CGP_ASSERT_OP(nodes[0],!=,nodes[1]);

    for(size_t n=0;n<2;++n){
        // assertions that node is active and
        // its own repr.
        const size_t  ni=nodes[n];
        CGP_ASSERT_OP(reprNode(ni),==,ni);
        CGP_ASSERT_OP(hasNode(ni),==,true);
    }



    // merge the two nodes
    nodeUfd_.merge(nodes[0],nodes[1]);
    const size_t newNodeRep    = reprNode(nodes[0]);
    const size_t notNewNodeRep =  (newNodeRep == nodes[0] ? nodes[1] : nodes[0] );

    const size_t  edgeSizeRep    = dynamicNodes_[newNodeRep].numberOfEdges();
    const size_t  edgeSizeNotRep = dynamicNodes_[notNewNodeRep].numberOfEdges();

    // the new region wich is the result of the merge
    NodeType & newFormedNode = dynamicNodes_[newNodeRep];

    // merge the edges of the nodes
    newFormedNode.mergeEdges(dynamicNodes_[notNewNodeRep]);
    CGP_ASSERT_OP(newFormedNode.numberOfEdges(),==,edgeSizeRep+edgeSizeNotRep-1);

    // delete the old region
    dynamicNodes_.erase(notNewNodeRep);

    // delete the edge which has been between those two regions
    // which we merge (since this edge is the one getting deleted)
    newFormedNode.eraseEdge(toDeleteEdgeIndex);
    dynamicEdges_.erase(toDeleteEdgeIndex);
    CGP_ASSERT_OP(newFormedNode.numberOfEdges(),==,edgeSizeRep+edgeSizeNotRep-2);


    // bevore processing with merging the edges we call the "merge" of the node maps
    // - we need to do this bevore any "merge" within the nodeMaps such that
    //   we can guarantee that the nodes maps are tidy when the edge-maps mergers
    //   are called
    for(size_t m=0;m<nodeMaps_.size();++m){
        nodeMaps_[m]->merge(nodes,newNodeRep);
    }



    // construct the "DoubleMap"
    // - if an vector in the map has a size >=2 
    //   this means that there are multiple edges
    //   between a pair of regions which needs to be merged
    // - furthermore all edges of the new formed Region
    //   are visited and if an edge uses the "notNewNodeRep" node
    //   index it will be relabed to newNodeRep
    DoubleMap doubleEdgeMap;
    this->fillDoubleMapAndRelabelNodes(newFormedNode,doubleEdgeMap,notNewNodeRep,newNodeRep);

    // loop over the double map
    // if an vector in the map has a size >=2 
    // this means that there are multiple edges
    // between a pair of regions which needs to be merged
    for( DoubleMap::const_iterator dIter = doubleEdgeMap.begin();dIter!=doubleEdgeMap.end();++dIter){

        // if this vector has a size >=2 this means we have multiple
        // edges between 2 regions
        // the 2 regions are encoded in the key (dIter->first)
        // but we do not need them here
        const std::vector<size_t> & edgeVec = dIter->second;
        if(edgeVec.size()>=2){

            // merge all these edges in the ufd and get the new representative
            const size_t newEdgeRep = edgeUfd_.multiMerge(edgeVec.front(),edgeVec.begin()+1,edgeVec.end());

            // delte all edges which are not needed any more
            //  - edgeVec.size() -1 edges will be deleted 
            //  - (all edges except the new representative "newEdgeRep")
            // furthermore  the edge-sets all nodes adjacent to the "newFormedNode"
            // must be visited since they might refere to nodes which are deleted /merged
            for(size_t td=0;td<edgeVec.size();++td){

                // index of the edge which is considered for deletion
                const size_t toMergeEdgeIndex = edgeVec[td];
                // delte this edge only if it is NOT the new representative edge
                if(toMergeEdgeIndex!=newEdgeRep){

                    // delete the edge from the new formed region
                    newFormedNode.edges_.erase(toMergeEdgeIndex);
                    CGP_ASSERT_OP(hasEdge(toMergeEdgeIndex),==,true);

                    // at least one of the nodes of the edge "toMergeEdgeIndex" must be the "newFormedNode"
                    //  - we want to get the nodes adjacent to the "newFormedNode"
                    CGP_ASSERT_OP(dynamicEdges_[toMergeEdgeIndex].hasNode(newNodeRep),==,true);
                    const size_t adjacentNodeIndex = dynamicEdges_[toMergeEdgeIndex].otherNode(newNodeRep);

                    dynamicNodes_[adjacentNodeIndex].eraseAndInsert(toMergeEdgeIndex,newEdgeRep);  
                    
                    // finaly delete the unneeded edge
                    dynamicEdges_.erase(toMergeEdgeIndex);

                }
            }

            // call merge for edge maps
            for(size_t m=0;m<edgeMaps_.size();++m){
                edgeMaps_[m]->merge(edgeVec,newEdgeRep);
            }
        } 
    }
    // call erase for edge maps
    for(size_t m=0;m<edgeMaps_.size();++m){
        edgeMaps_[m]->erase(toDeleteEdgeIndex,newNodeRep);
    }




    CGP_ASSERT_OP(dynamicNodes_.size(),==,preNumNodes-1);
    CGP_ASSERT_OP(nodeUfd_.numberOfSets(),==,preNumNodes-1);
    CGP_ASSERT_OP(this->numberOfNodes(),==,preNumNodes-1);
}




}



#endif //CGP2D_DYNAMIC_GRAPH_HXX