//*****************************************************************************
// Copyright 2018-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <boost/asio.hpp>

#include "he_seal_client.hpp"
#include "seal/he_seal_client.hpp"

namespace py = pybind11;

/* PYBIND11_MODULE(he_seal_client, m) {
  // void regclass_he_seal_client(py::module m) {
  py::class_<ngraph::runtime::he::HESealClient> he_seal_client(m,
                                                               "HESealClient");
  he_seal_client.doc() = "he_seal_client doc";

  he_seal_client.def(
      py::init<boost::asio::io_context&, const tcp::resolver::results_type&,
               std::vector<float>>());
} */

/* std::vector<double> modify(const std::vector<double>& input) {
  std::vector<double> output;

  std::transform(input.begin(), input.end(), std::back_inserter(output),
                 [](double x) -> double { return 2. * x; });

  // N.B. this is equivalent to (but there are also other ways to do the same)
  //
  // std::vector<double> output(input.size());
  //
  // for ( size_t i = 0 ; i < input.size() ; ++i )
  //   output[i] = 2. * input[i];

  return output;
}

// ----------------
// Python interface
// ----------------

namespace py = pybind11;

PYBIND11_MODULE(example, m) {
  m.doc() = "pybind11 example plugin";

  m.def("modify", &modify, "Multiply all entries of a list by 2.0");
} */