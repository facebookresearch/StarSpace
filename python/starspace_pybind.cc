#include <starspace.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(starwrap, m) {
	py::class_<starspace::Args>(m, "args")
		.def(py::init<>())
		.def_readwrite("input", &starspace::Args::trainFile)
		;
}
