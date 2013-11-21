


template<class CGP>
class CellIterator{

public:
	typedef typename CGP::LabelType LabelType;




private:
	const CGP & cgp_;
	LabelType cellLabel_;
	unsigned char cellType_;
};



template<class CGP,class CELL_TYPE,class VIA_CELL_TYPE>
class SelfAdjaceny{
	
};






/*
	CellIterator cIter = cgp.cell(2,0)
	
	++cIter will just get the  next cell of the current type		
*/