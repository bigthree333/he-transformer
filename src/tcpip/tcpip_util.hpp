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

#pragma once

#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <iostream>
using boost::asio::ip::tcp;

namespace ngraph {
namespace runtime {
namespace he {
namespace network {

// Connects to server at given hostname / port
// Returns socket of connection
tcp::socket connect_to_server(const std::string& hostname, const size_t port);

// Initializes a server at given port
void server_init(const size_t port, const size_t conn_limit = 100);

// Sends data to socket
// @param socket socket at which server is listening
// @param data data to send
// @param bytes number of bytes in data to send
void send_data(tcp::socket& socket, const std::string& data);

// Receives data
// @param socker socker at which we receive data from
// @return numbers of bytes of data received
void receive_data(tcp::socket& socket, void* data);

void session(tcp::socket sock);

// Closes connection at given socket
void close_socket(tcp::socket sock);

}  // namespace network
}  // namespace he
}  // namespace runtime
}  // namespace ngraph