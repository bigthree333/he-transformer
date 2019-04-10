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

#include <assert.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "test_util.hpp"

using namespace std;
using namespace ngraph;

vector<float> read_constant(const string filename) {
  string data = file_util::read_file_to_string(filename);
  istringstream iss(data);

  vector<string> constants;
  copy(istream_iterator<string>(iss), istream_iterator<string>(),
       back_inserter(constants));

  vector<float> res;
  for (const string& constant : constants) {
    res.push_back(atof(constant.c_str()));
  }
  return res;
}

vector<int> batched_argmax(const vector<float>& ys) {
  if (ys.size() % 10 != 0) {
    cout << "ys.size() must be a multiple of 10" << endl;
    exit(1);
  }
  vector<int> labels;
  const float* data = ys.data();
  size_t idx = 0;
  while (idx < ys.size()) {
    int label = distance(data + idx, max_element(data + idx, data + idx + 10));
    labels.push_back(label);
    idx += 10;
  }
  return labels;
}

float get_accuracy(const vector<float>& pre_sigmoid, const vector<float>& y) {
  assert(pre_sigmoid.size() % 10 == 0);
  size_t num_data = pre_sigmoid.size() / 10;

  size_t correct = 0;
  for (size_t i = 0; i < num_data; ++i) {
    vector<float> sub_vec(pre_sigmoid.begin() + i * 10,
                          pre_sigmoid.begin() + (i + 1) * 10);
    auto minmax = minmax_element(sub_vec.begin(), sub_vec.end());
    size_t prediction = minmax.second - sub_vec.begin();

    if (round(y[10 * i + prediction]) == 1) {
      correct++;
    }
  }
  return correct / float(num_data);
}

vector<float> read_binary_constant(const string filename, size_t num_elements) {
  ifstream infile;
  vector<float> values(num_elements);
  infile.open(filename, ios::in | ios::binary);

  infile.read(reinterpret_cast<char*>(&values[0]),
              num_elements * sizeof(float));
  infile.close();
  return values;
}

void write_binary_constant(const vector<float>& values, const string filename) {
  ofstream outfile(filename, ios::out | ios::binary);
  outfile.write(reinterpret_cast<const char*>(&values[0]),
                values.size() * sizeof(float));
  outfile.close();
}

vector<tuple<vector<shared_ptr<runtime::Tensor>>,
             vector<shared_ptr<runtime::Tensor>>>>
generate_plain_cipher_tensors(const vector<shared_ptr<Node>>& output,
                              const vector<shared_ptr<Node>>& input,
                              const runtime::Backend* backend,
                              bool consistent_type, bool skip_plain_plain) {
  auto he_backend = static_cast<const runtime::he::HEBackend*>(backend);

  using TupleOfInputOutputs =
      vector<tuple<vector<shared_ptr<runtime::Tensor>>,
                   vector<shared_ptr<runtime::Tensor>>>>;
  TupleOfInputOutputs ret;

  auto cipher_cipher = [&output, &input, &he_backend]() {
    vector<shared_ptr<runtime::Tensor>> result;
    for (auto elem : output) {
      auto output_tensor = he_backend->create_cipher_tensor(
          elem->get_element_type(), elem->get_shape());
      result.push_back(output_tensor);
    }
    vector<shared_ptr<runtime::Tensor>> argument;
    for (auto elem : input) {
      auto input_tensor = he_backend->create_cipher_tensor(
          elem->get_element_type(), elem->get_shape());
      argument.push_back(input_tensor);
    }
    return make_tuple(result, argument);
  };

  auto plain_plain = [&output, &input, &he_backend]() {
    vector<shared_ptr<runtime::Tensor>> result;
    for (auto elem : output) {
      auto output_tensor = he_backend->create_plain_tensor(
          elem->get_element_type(), elem->get_shape());
      result.push_back(output_tensor);
    }
    vector<shared_ptr<runtime::Tensor>> argument;
    for (auto elem : input) {
      auto input_tensor = he_backend->create_plain_tensor(
          elem->get_element_type(), elem->get_shape());
      argument.push_back(input_tensor);
    }
    return make_tuple(result, argument);
  };
  auto alternate_cipher = [&output, &input, &he_backend](size_t mod) {
    vector<shared_ptr<runtime::Tensor>> result;
    for (auto elem : output) {
      auto output_tensor = he_backend->create_cipher_tensor(
          elem->get_element_type(), elem->get_shape());
      result.push_back(output_tensor);
    }
    vector<shared_ptr<runtime::Tensor>> argument;
    for (size_t i = 0; i < input.size(); ++i) {
      auto elem = input[i];
      if (i % 2 == mod) {
        auto input_tensor = he_backend->create_plain_tensor(
            elem->get_element_type(), elem->get_shape());
        argument.push_back(input_tensor);
      } else {
        auto input_tensor = he_backend->create_cipher_tensor(
            elem->get_element_type(), elem->get_shape());
        argument.push_back(input_tensor);
      }
    }
    return make_tuple(result, argument);
  };
  auto plain_cipher_cipher = [&output, &input, &he_backend,
                              &alternate_cipher]() {
    return alternate_cipher(0);
  };
  auto cipher_plain_cipher = [&output, &input, &he_backend,
                              &alternate_cipher]() {
    return alternate_cipher(1);
  };

  if (he_backend != nullptr) {
    ret.push_back(cipher_cipher());
    if (!skip_plain_plain) {
      ret.push_back(plain_plain());
    }
    if (!consistent_type) {
      ret.push_back(plain_cipher_cipher());
    }
    if (input.size() >= 2 && !consistent_type) {
      ret.push_back(cipher_plain_cipher());
    }
  }

  return ret;
}

// Runs each op in a file generated by nbench statistics
void nbench_summary_perf(const runtime::he::HEBackend* he_backend,
                         const std::string& filename) {
  std::ifstream file(filename);
  NGRAPH_INFO << "Filename " << filename;
  if (file.is_open()) {
    std::string line;
    while (getline(file, line)) {
      size_t left_brace_pos = line.find("{");
      size_t right_brace_pos = line.find("}");
      size_t colon_pos = line.find(":");
      size_t ops_pos = line.find("ops");

      std::string op = line.substr(0, left_brace_pos);
      std::string shape_str =
          line.substr(left_brace_pos + 1, right_brace_pos - left_brace_pos - 1);
      std::string count =
          line.substr(right_brace_pos + 3, ops_pos - 4 - right_brace_pos);

      NGRAPH_INFO << "op " << op;
      NGRAPH_INFO << "shape_str " << shape_str;
      NGRAPH_INFO << "count " << count;

      std::vector<size_t> shape_dims{};
      size_t comma_pos = shape_str.find(",");
      while (comma_pos != std::string::npos) {
        comma_pos = shape_str.find(",");
        auto sub = shape_str.substr(0, comma_pos);
        size_t shape_dim = std::stoi(sub);
        shape_dims.emplace_back(shape_dim);
        shape_str.erase(0, comma_pos + 1);
      }
      NGRAPH_INFO << "Shape";
      for (const auto& elem : shape_dims) {
        NGRAPH_INFO << elem;
      }
      /*
            Shape shape_a{1, 3, 2, 2};
            auto A = make_shared<op::Parameter>(element::f32, shape_a);
            Shape shape_b{};
            auto B = make_shared<op::Parameter>(element::f32, shape_b);
            Shape shape_r{1, 1, 4, 4};
            CoordinateDiff padding_below{0, -1, 1, 1};
            CoordinateDiff padding_above{0, -1, 1, 1};
            auto f = make_shared<Function>(
                make_shared<op::Pad>(A, B, padding_below, padding_above),
                ParameterVector{A, B});

            auto backend = runtime::Backend::create("${BACKEND_NAME}");
            auto he_backend =
         static_cast<runtime::he::HEBackend*>(backend.get());

            // Create some tensors for input/output
            auto a = he_backend->create_cipher_tensor(element::f32, shape_a);
            // clang-format off
          copy_data(a, test::NDArray<float, 4>(
              {
                  {
                      {
                          {0.0f, 0.0f},
                          {0.0f, 0.0f}
                      },
                      {
                          {1.0f, 1.0f},
                          {1.0f, 1.0f}
                      },
                      {
                          {2.0f, 2.0f},
                          {2.0f, 2.0f}
                      }
                  }
              }).get_vector());
            // clang-format on

            auto b = he_backend->create_cipher_tensor(element::f32, shape_b);
            copy_data(b, vector<float>{42});

            auto result = he_backend->create_cipher_tensor(element::f32,
         shape_r);

            auto handle = backend->compile(f);
            handle->call_with_validate({result}, {a, b}); */
    }
    file.close();
  }
}
