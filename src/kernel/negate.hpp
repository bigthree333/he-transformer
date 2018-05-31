/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <memory>
#include <vector>

namespace ngraph
{
    namespace element
    {
        class Type;
    }
    namespace runtime
    {
        namespace he
        {
            class HEBackend;

            namespace kernel
            {
                void negate(const std::vector<std::shared_ptr<runtime::he::HECiphertext>>& arg,
                            std::vector<std::shared_ptr<runtime::he::HECiphertext>>& out,
                            const element::Type& type,
                            std::shared_ptr<runtime::he::HEBackend> he_backend,
                            size_t count);

                void negate(const std::vector<std::shared_ptr<runtime::he::HEPlaintext>>& arg,
                            std::vector<std::shared_ptr<runtime::he::HEPlaintext>>& out,
                            const element::Type& type,
                            std::shared_ptr<runtime::he::HEBackend> he_backend,
                            size_t count);

                void scalar_negate(const std::shared_ptr<runtime::he::HECiphertext>& arg,
                                   std::shared_ptr<runtime::he::HECiphertext>& out,
                                   const element::Type& type,
                                   std::shared_ptr<runtime::he::HEBackend> he_backend);

                void scalar_negate(const std::shared_ptr<runtime::he::HEPlaintext>& arg,
                                   std::shared_ptr<runtime::he::HEPlaintext>& out,
                                   const element::Type& type,
                                   std::shared_ptr<runtime::he::HEBackend> he_backend);
            }
        }
    }
}