// Copyright 2026 Zilliz
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

#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <arrow/buffer.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/io/memory.h>
#include <arrow/util/key_value_metadata.h>

namespace milvus_storage::test {

// Fake Talon coordinator/worker that records incoming data-plane message types
// and replies with a deterministic fill byte (distinct from MetadataInputFile's 'x').
class RecordingTalonServer {
  public:
  explicit RecordingTalonServer(uint8_t fill_byte = 0x5a, bool fail_reads = false)
      : fill_byte_(fill_byte), fail_reads_(fail_reads) {
    listener_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listener_ < 0) {
      throw std::runtime_error("recording Talon server socket failed");
    }
    const int one = 1;
    (void)::setsockopt(listener_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::bind(listener_, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0 ||
        ::listen(listener_, 16) != 0) {
      ::close(listener_);
      throw std::runtime_error("recording Talon server bind/listen failed");
    }
    socklen_t size = sizeof(address);
    if (::getsockname(listener_, reinterpret_cast<sockaddr*>(&address), &size) != 0) {
      ::close(listener_);
      throw std::runtime_error("recording Talon server getsockname failed");
    }
    port_ = ntohs(address.sin_port);
    thread_ = std::thread([this] { Serve(); });
  }

  ~RecordingTalonServer() {
    (void)::shutdown(listener_, SHUT_RDWR);
    thread_.join();
    ::close(listener_);
  }

  RecordingTalonServer(const RecordingTalonServer&) = delete;
  RecordingTalonServer& operator=(const RecordingTalonServer&) = delete;

  [[nodiscard]] std::string Address() const { return "127.0.0.1:" + std::to_string(port_); }

  struct RecordedRead {
    uint8_t msg_type;
    uint64_t offset;
    uint64_t length;
  };

  [[nodiscard]] std::vector<RecordedRead> Reads() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return reads_;
  }

  private:
  static bool RecvAll(int fd, uint8_t* data, size_t size) {
    size_t offset = 0;
    while (offset < size) {
      const auto received = ::recv(fd, data + offset, size - offset, 0);
      if (received <= 0) {
        return false;
      }
      offset += static_cast<size_t>(received);
    }
    return true;
  }

  static bool SendAll(int fd, const uint8_t* data, size_t size) {
    size_t offset = 0;
    while (offset < size) {
#ifdef MSG_NOSIGNAL
      const auto sent = ::send(fd, data + offset, size - offset, MSG_NOSIGNAL);
#else
      const auto sent = ::send(fd, data + offset, size - offset, 0);
#endif
      if (sent <= 0) {
        return false;
      }
      offset += static_cast<size_t>(sent);
    }
    return true;
  }

  static bool ReadU32LE(const std::vector<uint8_t>& buf, size_t& pos, uint32_t& out) {
    if (pos + 4 > buf.size()) {
      return false;
    }
    out = uint32_t(buf[pos]) | (uint32_t(buf[pos + 1]) << 8) | (uint32_t(buf[pos + 2]) << 16) |
          (uint32_t(buf[pos + 3]) << 24);
    pos += 4;
    return true;
  }

  static bool ReadU64LE(const std::vector<uint8_t>& buf, size_t& pos, uint64_t& out) {
    if (pos + 8 > buf.size()) {
      return false;
    }
    out = 0;
    for (int i = 0; i < 8; ++i) {
      out |= uint64_t(buf[pos + i]) << (8 * i);
    }
    pos += 8;
    return true;
  }

  static bool SkipBincodeString(const std::vector<uint8_t>& buf, size_t& pos) {
    uint64_t len = 0;
    if (!ReadU64LE(buf, pos, len) || len > buf.size() - pos) {
      return false;
    }
    pos += static_cast<size_t>(len);
    return true;
  }

  // Decodes offset/length from VersionedRangeRequest (msg_type 10/12) or
  // NamespaceScopedVersionedRange (msg_type 11/13).
  static bool DecodeRangeRequest(
      uint8_t msg_type, const std::vector<uint8_t>& payload, size_t pos, uint64_t& offset, uint64_t& length) {
    if (msg_type == 11 || msg_type == 13) {
      uint32_t ns_variant = 0;
      if (!ReadU32LE(payload, pos, ns_variant)) {
        return false;
      }
      if (ns_variant == 0 && !SkipBincodeString(payload, pos)) {
        return false;
      }
    }
    uint32_t backend = 0;
    if (!ReadU32LE(payload, pos, backend)) {
      return false;
    }
    if (!SkipBincodeString(payload, pos) || !SkipBincodeString(payload, pos)) {
      return false;
    }
    return ReadU64LE(payload, pos, offset) && ReadU64LE(payload, pos, length);
  }

  void Serve() {
    for (;;) {
      const int connection = ::accept(listener_, nullptr, nullptr);
      if (connection < 0) {
        return;
      }
      const timeval timeout{5, 0};
      (void)::setsockopt(connection, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
      (void)::setsockopt(connection, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
#ifdef SO_NOSIGPIPE
      const int one = 1;
      (void)::setsockopt(connection, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
      std::array<uint8_t, 16> header{};
      if (!RecvAll(connection, header.data(), header.size())) {
        ::close(connection);
        continue;
      }
      uint32_t payload_size = 0;
      std::memcpy(&payload_size, header.data() + 12, sizeof(payload_size));
      payload_size = ntohl(payload_size);
      if (payload_size > 32768) {
        ::close(connection);
        continue;
      }
      std::vector<uint8_t> payload(payload_size);
      if (payload_size > 0 && !RecvAll(connection, payload.data(), payload.size())) {
        ::close(connection);
        continue;
      }
      size_t metadata_end = 0;
      if (header[2] >= 2) {
        if (payload.size() < 2) {
          ::close(connection);
          continue;
        }
        metadata_end = 2 + (static_cast<size_t>(payload[0]) << 8) + payload[1];
      }
      const bool membership = header[3] == 0 && payload.size() >= metadata_end + 3 && payload[metadata_end + 2] == 4;
      std::vector<uint8_t> body;
      if (membership) {
        body = {6, 0, 5, 0, 0, 0};
        auto append_u64 = [&body](uint64_t value) {
          for (int i = 0; i < 8; ++i) {
            body.push_back(static_cast<uint8_t>(value >> (8 * i)));
          }
        };
        auto append_string = [&body, &append_u64](const std::string& value) {
          append_u64(value.size());
          body.insert(body.end(), value.begin(), value.end());
        };
        append_u64(1);
        append_u64(1);
        append_u64(30000);
        append_u64(1);
        append_string("worker");
        body.insert(body.end(), {0, 0, 2, 0, 0, 0});
        append_string("instance");
        append_string(Address());
      } else {
        uint64_t offset = 0;
        uint64_t length = 0;
        if (header[3] < 10 || header[3] > 13 || !DecodeRangeRequest(header[3], payload, metadata_end, offset, length) ||
            length > 2 * 1024 * 1024) {
          ::close(connection);
          continue;
        }
        {
          std::lock_guard<std::mutex> lock(mutex_);
          reads_.push_back({header[3], offset, length});
        }
        header[3] = 2;  // Raw GetRange response.
        if (fail_reads_) {
          header[5] = 2;  // ERROR flag. The protocol accepts a legacy text error body.
          const std::string error = "exact reads require small-object storage to be enabled";
          body.assign(error.begin(), error.end());
        } else {
          body.assign(static_cast<size_t>(length), fill_byte_);
        }
      }
      std::vector<uint8_t> response(header.begin(), header.end());
      const uint32_t response_size = htonl(static_cast<uint32_t>(body.size()));
      std::memcpy(response.data() + 12, &response_size, sizeof(response_size));
      response.insert(response.end(), body.begin(), body.end());
      (void)SendAll(connection, response.data(), response.size());
      ::close(connection);
    }
  }

  int listener_ = -1;
  int port_ = 0;
  uint8_t fill_byte_;
  bool fail_reads_;
  std::thread thread_;
  mutable std::mutex mutex_;
  std::vector<RecordedRead> reads_;
};

// Expose writer-generated fixtures as versioned objects, including for opens
// that resolve size and ETag lazily. Each handle owns its own cursor.
class TalonFixtureOrigin final : public arrow::fs::SubTreeFileSystem {
  public:
  explicit TalonFixtureOrigin(const std::string& root)
      : arrow::fs::SubTreeFileSystem(root, std::make_shared<arrow::fs::LocalFileSystem>()) {}

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const std::string& path) override {
    ARROW_ASSIGN_OR_RAISE(auto file, SubTreeFileSystem::OpenInputFile(path));
    ARROW_ASSIGN_OR_RAISE(auto size, file->GetSize());
    ARROW_ASSIGN_OR_RAISE(auto data, file->ReadAt(0, size));
    return std::make_shared<VersionedInputFile>(std::move(data));
  }

  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const arrow::fs::FileInfo& info) override {
    return OpenInputFile(info.path());
  }

  private:
  class VersionedInputFile final : public arrow::io::BufferReader {
 public:
    using arrow::io::BufferReader::BufferReader;

    arrow::Result<std::shared_ptr<const arrow::KeyValueMetadata>> ReadMetadata() override {
      return std::shared_ptr<const arrow::KeyValueMetadata>(arrow::key_value_metadata({"ETag"}, {"fixture-etag"}));
    }
  };
};

}  // namespace milvus_storage::test
