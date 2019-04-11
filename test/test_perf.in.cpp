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

#include <chrono>
#include <fstream>
#include <thread>
#include "he_backend.hpp"
#include "ngraph/ngraph.hpp"
#include "test_util.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, perf_add) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  size_t N{10000};

  Shape shape{N};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Add>(a, b);
  auto f = make_shared<Function>(t, ParameterVector{a, b});

  vector<float> vec_a(N, 0);
  vector<float> vec_b(N, 0);
  vector<float> vec_result(N, 0);

  for (size_t i = 0; i < N; ++i) {
    vec_a[i] = i;
    vec_b[i] = i;
    vec_result[i] = vec_a[i] + vec_b[i];
  }

  // Create some tensors for input/output
  auto tensors_list =
      generate_plain_cipher_tensors({t}, {a, b}, backend.get(), false, true);

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto t_a = inputs[0];
    auto t_b = inputs[1];
    auto t_result = results[0];

    copy_data(t_a, vec_a);
    copy_data(t_b, vec_b);
    auto handle = backend->compile(f);
    handle->call({t_result}, {t_a, t_b});
  }
}

NGRAPH_TEST(${BACKEND_NAME}, perf_square) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  size_t N{10000};

  Shape shape{N};
  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Multiply>(a, b);
  auto f = make_shared<Function>(t, ParameterVector{a, b});

  vector<float> vec_a(N, 0);
  vector<float> vec_b(N, 0);
  vector<float> vec_result(N, 0);

  for (size_t i = 0; i < N; ++i) {
    vec_a[i] = i + 1;
    vec_b[i] = i + 1;
    vec_result[i] = vec_a[i] * vec_b[i];
  }

  // Create some tensors for input/output
  auto tensors_list =
      generate_plain_cipher_tensors({t}, {a, b}, backend.get(), false, true);

  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto t_a = inputs[0];
    auto t_b = inputs[1];
    auto t_result = results[0];

    copy_data(t_a, vec_a);
    copy_data(t_b, vec_b);
    auto handle = backend->compile(f);
    handle->call({t_result}, {t_a, t_b});
  }
}

NGRAPH_TEST(${BACKEND_NAME}, nbench_file) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<runtime::he::HEBackend*>(backend.get());

  ifstream in("../examples/MobileNetV2/tf_function_ngraph_cluster_0.json");
  shared_ptr<Function> func = deserialize(in);

  nbench_summary_perf(he_backend, func);
}

NGRAPH_TEST(${BACKEND_NAME}, create_cifar10_tensor) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<runtime::he::HEBackend*>(backend.get());

  Shape shape{32, 32, 3};
  auto a = he_backend->create_cipher_tensor(element::f32, shape);
  std::vector<float> vals(32 * 32 * 3, 1.23);
  copy_data(a, vals);

  // Wait so we can measure memory accurately
  std::this_thread::sleep_for(std::chrono::seconds(10));
}

NGRAPH_TEST(${BACKEND_NAME}, create_imagenet_tensor) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<runtime::he::HEBackend*>(backend.get());

  Shape shape{224, 224, 3};
  auto a = he_backend->create_cipher_tensor(element::f32, shape);

  std::vector<float> vals(224 * 224 * 3, 1.23);
  copy_data(a, vals);

  // Wait so we can measure memory accurately
  std::this_thread::sleep_for(std::chrono::seconds(10));
}
