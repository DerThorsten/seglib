#define PY_ARRAY_UNIQUE_SYMBOL superimg_PyArray_API
#define NO_IMPORT_ARRAY


#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>

#include "seglib/cgp2d/cgp2d.hxx"
#include "seglib/cgp2d/cgp2d_python.hxx"
#include "py_cell_visitor.hxx"

namespace cgp2d {

namespace python = boost::python;

void export_cell1()
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
    typedef CgpType::Cell1 Cell1Type;
    typedef CgpType::Cells1 Cell1VectorType;

    // cells
    python::class_<Cell1Type>("Cell1",python::init<>())
        .def(CellTypeSuite<Cell1Type>())
    ;
}

} // namespace vigra

