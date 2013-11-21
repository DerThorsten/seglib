#define PY_ARRAY_UNIQUE_SYMBOL phist_PyArray_API

//#include <Python.h>
#include <boost/python.hpp>
#include <vigra/numpy_array_converters.hxx>

#include "seglib/histogram/histogram_python.hxx"


#include "py_histogram.hxx"
#include "py_label_histogram.hxx"
#include "py_histogram_gradient.hxx"

BOOST_PYTHON_MODULE_INIT(_histogram) {
    //using namespace boost::python;
    //using namespace vigra;


    import_array();
    //boost::python::array::set_module_and_type("numpy", "ndarray");
    vigra::import_vigranumpy();


    histogram::export_histogram();
    histogram::export_label_histogram();
    histogram::export_histogram_gradient();
}
