#include <starspace.h>
#include <matrix.h>
#include <model.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(starwrap, m) {
	py::class_<starspace::Args, std::shared_ptr<starspace::Args>>(m, "args")
		.def(py::init<>())
		.def_readwrite("trainFile", &starspace::Args::trainFile)
		.def_readwrite("validationFile", &starspace::Args::validationFile)
		.def_readwrite("testFile", &starspace::Args::testFile)
		.def_readwrite("predictionFile", &starspace::Args::predictionFile)
		.def_readwrite("model", &starspace::Args::model)
		.def_readwrite("initModel", &starspace::Args::initModel)
		.def_readwrite("fileFormat", &starspace::Args::fileFormat)
		.def_readwrite("label", &starspace::Args::label)
		.def_readwrite("basedoc", &starspace::Args::basedoc)
		.def_readwrite("loss", &starspace::Args::loss)
		.def_readwrite("similarity", &starspace::Args::similarity)
		.def_readwrite("lr", &starspace::Args::lr)
		.def_readwrite("termLr", &starspace::Args::termLr)
		.def_readwrite("norm", &starspace::Args::norm)
		.def_readwrite("margin", &starspace::Args::margin)
		.def_readwrite("initRandSd", &starspace::Args::initRandSd)
		.def_readwrite("p", &starspace::Args::p)
		.def_readwrite("dropoutLHS", &starspace::Args::dropoutLHS)
		.def_readwrite("dropoutRHS", &starspace::Args::dropoutRHS)
		.def_readwrite("wordWeight", &starspace::Args::wordWeight)
		.def_readwrite("dim", &starspace::Args::dim)
		.def_readwrite("epoch", &starspace::Args::epoch)
		.def_readwrite("ws", &starspace::Args::ws)
		.def_readwrite("maxTrainTime", &starspace::Args::maxTrainTime)
		.def_readwrite("validationPatience", &starspace::Args::validationPatience)
		.def_readwrite("thread", &starspace::Args::thread)
		.def_readwrite("maxNegSamples", &starspace::Args::maxNegSamples)
		.def_readwrite("negSearchLimit", &starspace::Args::negSearchLimit)
		.def_readwrite("minCount", &starspace::Args::minCount)
		.def_readwrite("minCountLabel", &starspace::Args::minCountLabel)
		.def_readwrite("bucket", &starspace::Args::bucket)
		.def_readwrite("ngrams", &starspace::Args::ngrams)
		.def_readwrite("trainMode", &starspace::Args::trainMode)
		.def_readwrite("K", &starspace::Args::K)
		.def_readwrite("batchSize", &starspace::Args::batchSize)
		.def_readwrite("verbose", &starspace::Args::verbose)
		.def_readwrite("debug", &starspace::Args::debug)
		.def_readwrite("adagrad", &starspace::Args::adagrad)
		.def_readwrite("isTrain", &starspace::Args::isTrain)
		.def_readwrite("normalizeText", &starspace::Args::normalizeText)
		.def_readwrite("saveEveryEpoch", &starspace::Args::saveEveryEpoch)
		.def_readwrite("saveTempModel", &starspace::Args::saveTempModel)
		.def_readwrite("shareEmb", &starspace::Args::shareEmb)
		.def_readwrite("useWeight", &starspace::Args::useWeight)
		.def_readwrite("trainWord", &starspace::Args::trainWord)
		.def_readwrite("excludeLHS", &starspace::Args::excludeLHS)
		;

	py::class_<starspace::Matrix <starspace::Real>>(m, "Matrix", py::buffer_protocol())
		.def_buffer([](starspace::Matrix <starspace::Real> &m) -> py::buffer_info {
			return py::buffer_info(
				m.matrix.data().begin(),							/* Pointer to buffer */
				sizeof(starspace::Real),							/* Size of one scalar */
				py::format_descriptor<starspace::Real>::format(), 	/* Python struct-style format descriptor */
				2,									    			/* Number of dimensions */
				{ m.numRows(), m.numCols() },				        /* Buffer dimensions */
				{ sizeof(starspace::Real) * m.numCols(),			/* Strides (in bytes) for each index */
				  sizeof(starspace::Real) }
			);
		}
	);

	py::class_<starspace::StarSpace>(m, "starSpace")
		.def(py::init<std::shared_ptr<starspace::Args>>())
		.def("init", &starspace::StarSpace::init)
		.def("initFromTsv", &starspace::StarSpace::initFromTsv)
		.def("initFromSavedModel", &starspace::StarSpace::initFromSavedModel)

		.def("train", &starspace::StarSpace::train)
		.def("evaluate", &starspace::StarSpace::evaluate)

		.def("getDocVector", &starspace::StarSpace::getDocVector)

		.def("nearestNeighbor", &starspace::StarSpace::nearestNeighbor)
		.def("predictTags", &starspace::StarSpace::predictTags)

		.def("saveModel", &starspace::StarSpace::saveModel)
		.def("saveModelTsv", &starspace::StarSpace::saveModelTsv)
		.def("loadBaseDocs", &starspace::StarSpace::loadBaseDocs)
		;
}
