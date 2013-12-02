#define PY_ARRAY_UNIQUE_SYMBOL superimg_PyArray_API

//#include <Python.h>
#include <boost/python.hpp>
#include <vigra/numpy_array_converters.hxx>



// basic cgp related stuff

#include "cells/py_cell0.hxx"
#include "cells/py_cell1.hxx"
#include "cells/py_cell2.hxx"

#include "cells/py_cell0vec.hxx"
#include "cells/py_cell1vec.hxx"
#include "cells/py_cell2vec.hxx"

#include "py_cgp2d.hxx"


// misc
#include "misc/py_segcompare.hxx"
#include "misc/py_merge.hxx"

BOOST_PYTHON_MODULE_INIT(_cgp2d) {
    //using namespace boost::python;
    //using namespace vigra;


    import_array();
    //boost::python::array::set_module_and_type("numpy", "ndarray");

    vigra::import_vigranumpy();



    // baisc cgp related data structures
    cgp2d::export_cgp2d();
    cgp2d::export_cell0();
    cgp2d::export_cell1();
    cgp2d::export_cell2();

    cgp2d::export_cell0vec();
    cgp2d::export_cell1vec();
    cgp2d::export_cell2vec();


    // image processing related functions and classes 
    cgp2d::export_segcompare();
    cgp2d::export_merge();

}
