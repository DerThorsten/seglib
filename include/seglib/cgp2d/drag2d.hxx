#ifndef CGP2D_DRAG_HXX
#define CGP2D_DRAG_HXX

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

namespace cgp2d{


template<class COORDINATE_TYPE,class LABEL_TYPE>
class Geometry;




template<class COORDINATE_TYPE,class LABEL_TYPE,int CELLTYPE>
class Cell;


template<class COORDINATE_TYPE, class LABEL_TYPE>
class Cgp;



struct DynamicEdge{
	DynamicEdge(){}

	DynamicEdge(const size_t cgpLabel,const size_t rl0,const size_t rl1){
		CGP_ASSERT_OP(rl0,>,0);
		CGP_ASSERT_OP(rl1,>,0);
		CGP_ASSERT_OP(rl0,!=,rl1);

		cgpEdges_.insert(cgpLabel);
		dRegs_[0]=std::min(rl0,rl1);
		dRegs_[1]=std::max(rl0,rl1);

		rRegs_[0]=std::min(rl0,rl1);
		rRegs_[1]=std::max(rl0,rl1);
	}

	std::set<size_t> cgpEdges_;
	size_t dRegs_[2];
	size_t rRegs_[2];
};



struct DynamicNodes{
	DynamicNodes(){}
	DynamicNodes(const size_t cgpLabel){
		cgpLabels_.insert(cgpLabel);
	}

	std::set<size_t> cgpLabels_;
};





template<class CGP>
class DynamicRag{
public:


    typedef std::map< size_t , DynamicEdge   >  EdgeMapType;
    typedef std::map< size_t , DynamicNodes >  NodesMapType;

    DynamicRag(const CGP & cgp)
    :   cgp_(cgp){

    	for(size_t ri=0;ri<cgp.numCells(2);++ri){
    		const size_t rl=ri+1;
    		nodes_[rl]=DynamicNodes(rl);
    	}

    	std::map<size_t ,size_t>  keyToEdge;

    	size_t edgeCounter=1;
    	for(size_t bi=0;bi<cgp.numCells(1);++bi){
    		const size_t bl=bi+1;
    		size_t rl0=cgp.bound(1,bi,0);
    		size_t rl1=cgp.bound(1,bi,1);
    		if (rl1<rl0){std::swap(rl0,rl1);}
    		const size_t key = rl0 + rl1 * (cgp.numCells(2)+1);

    		if(keyToEdge.find(key)==keyToEdge.end()){
    			keyToEdge[key]=edgeCounter;
    			edges_[edgeCounter]=DynamicEdge(bl,rl0,rl1);
    			++edgeCounter;
    		}
    		else{
    			const size_t edgeLabel = keyToEdge[key];
    			edges_[edgeLabel].cgpEdges_.insert(bl);
    		}
    	}
    }

    size_t numberOfNodes()const{
    	return nodes_.size();
    }

    size_t numberOfEdges()const{
    	return edges_.size();
    }

    void removeEdge(const size_t edgeLabel ){
        CGP_ASSERT_OP( bool(edges_.find(edgeLabel)!=edges_.end()),==,true );
    }


    const EdgeMapType &  edgeMap()const{
        return edges_;
    }
    //EdgeMapType & edgeMap(){
    //    return edges_;
    //}
    const NodesMapType & nodeMap()const{
        return nodes_;
    }
    //NodesMapType & nodeMap(){
    //    return nodes_;
    //}



private:
    const CGP & cgp_;



    EdgeMapType  edges_;
    NodesMapType  nodes_;

};


} /* namespace cgp2d */

#endif /* CGP2D_DRAG_HXX */