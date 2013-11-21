include_directories(
    ${PYTHON_INCLUDE_PATH}
    ${PYTHON_NUMPY_INCLUDE_DIR}
)


add_library(_histogram SHARED
    py_histogram.cxx
    py_label_histogram.cxx
    py_histogram_gradient.cxx
    py_main.cxx
)
target_link_libraries(_histogram
    ${Boost_PYTHON_LIBRARIES}
    ${VIGRA_IMPEX_LIBRARY}
    ${VIGRA_NUMPY_CORE_LIBRARY}
    ${VIGRA_NUMPY_IMPEX_LIBRARY}
)
set_target_properties(_histogram PROPERTIES
    PREFIX ""
)
get_property(location TARGET _histogram PROPERTY LOCATION)
add_custom_command(TARGET _histogram
    POST_BUILD        
    COMMAND ${CMAKE_COMMAND} -E copy_if_different  
        ${location}
        ${CMAKE_SOURCE_DIR}"/seglibpython/seglib/histogram"
)

