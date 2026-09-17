// Copyright 2026 Zilliz
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <future>
#include <stdexcept>
#include <thread>
#include <vector>
#include <opentelemetry/exporters/memory/in_memory_span_exporter.h>
#include <opentelemetry/sdk/trace/simple_processor.h>
#include <opentelemetry/sdk/trace/tracer_provider.h>
#include <opentelemetry/sdk/trace/samplers/parent.h>
#include <opentelemetry/sdk/trace/samplers/always_on.h>
#include <opentelemetry/sdk/trace/samplers/always_off.h>
#include "milvus-storage/tracing.h"
#include "talon/talon_bridge.h"
#include "tracing/runtime.h"

namespace milvus_storage::tracing {
namespace {
namespace ot = opentelemetry::trace;
namespace sdk = opentelemetry::sdk::trace;
using Memory = opentelemetry::exporter::memory::InMemorySpanData;

// A one-request TCP peer uses the pinned Talon control protocol, including v2
// request metadata and the unchanged bincode ObjectStat response body.
class StatPeer {
  public:
  StatPeer() : release_(release_promise_.get_future()) {
    socket_ = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (socket_ < 0 || bind(socket_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
        listen(socket_, 1) != 0) {
      throw std::runtime_error("failed to start Talon test peer");
    }
    socklen_t size = sizeof(address);
    getsockname(socket_, reinterpret_cast<sockaddr*>(&address), &size);
    address_ = "127.0.0.1:" + std::to_string(ntohs(address.sin_port));
    thread_ = std::thread([this] { Serve(); });
  }
  ~StatPeer() {
    Release();
    shutdown(socket_, SHUT_RDWR);
    if (thread_.joinable())
      thread_.join();
    close(socket_);
  }
  const std::string& address() const { return address_; }
  bool Wait() { return received_.get_future().wait_for(std::chrono::seconds(10)) == std::future_status::ready; }
  void Release() {
    if (!released_.exchange(true))
      release_promise_.set_value();
  }
  uint8_t version = 0;
  std::string parent, state;

  private:
  static bool Receive(int fd, uint8_t* out, size_t length) {
    while (length) {
      auto n = recv(fd, out, length, 0);
      if (n <= 0)
        return false;
      out += n;
      length -= n;
    }
    return true;
  }
  void Serve() {
    int client = accept(socket_, nullptr, nullptr);
    if (client < 0)
      return;
    timeval timeout{5, 0};
    setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    std::array<uint8_t, 16> header{};
    if (Receive(client, header.data(), header.size())) {
      uint32_t size;
      std::memcpy(&size, header.data() + 12, 4);
      size = ntohl(size);
      if (size <= 4096) {
        std::vector<uint8_t> body(size);
        if (Receive(client, body.data(), body.size())) {
          version = header[2];
          if (version == 2 && body.size() >= 2) {
            const size_t end = 2 + (size_t(body[0]) << 8) + body[1];
            for (size_t at = 2; end <= body.size() && at + 3 <= end;) {
              auto tag = body[at];
              size_t len = (size_t(body[at + 1]) << 8) + body[at + 2];
              at += 3;
              if (len > end - at)
                break;
              if (tag == 1)
                parent.assign(reinterpret_cast<char*>(body.data() + at), len);
              if (tag == 2)
                state.assign(reinterpret_cast<char*>(body.data() + at), len);
              at += len;
            }
          }
          received_.set_value();
          release_.wait();
          // schema=2, ObjectStat variant=11, size=7, version="v1" (little endian).
          const std::array<uint8_t, 24> reply{2, 0, 11, 0, 0, 0, 7, 0, 0, 0, 0,   0,
                                              0, 0, 2,  0, 0, 0, 0, 0, 0, 0, 'v', '1'};
          const auto length = htonl(reply.size());
          std::memcpy(header.data() + 12, &length, 4);
          std::vector<uint8_t> frame(header.begin(), header.end());
          frame.insert(frame.end(), reply.begin(), reply.end());
          (void)send(client, frame.data(), frame.size(), 0);
        }
      }
    }
    close(client);
  }
  int socket_;
  std::string address_;
  std::thread thread_;
  std::promise<void> received_, release_promise_;
  std::future<void> release_;
  std::atomic<bool> released_{false};
};

class TalonTracingTest : public testing::Test {
  protected:
  static void SetUpTestSuite() { setenv("TALON_TELEMETRY_V2_ENDPOINTS", "*", 1); }
  static ProviderPtr MakeProvider(std::shared_ptr<Memory>& data, bool reject_remote = false) {
    auto exporter = std::make_unique<opentelemetry::exporter::memory::InMemorySpanExporter>(512);
    data = exporter->GetData();
    std::shared_ptr<sdk::Sampler> remote_sampled = reject_remote
                                                       ? std::shared_ptr<sdk::Sampler>(new sdk::AlwaysOffSampler())
                                                       : std::shared_ptr<sdk::Sampler>(new sdk::AlwaysOnSampler());
    return ProviderPtr(new sdk::TracerProvider(
        std::make_unique<sdk::SimpleSpanProcessor>(std::move(exporter)),
        opentelemetry::sdk::resource::Resource::Create({}),
        std::make_unique<sdk::ParentBasedSampler>(std::make_unique<sdk::AlwaysOnSampler>(), remote_sampled)));
  }
  static TraceParent Parent(uint8_t id = 1, bool sampled = true) {
    TraceParent parent;
    parent.trace_id[0] = id;
    parent.span_id[0] = id;
    parent.trace_flags = sampled ? 1 : 0;
    parent.tracestate = "vendor=value";
    return parent;
  }
  static ot::TraceId TraceId(uint8_t id = 1) {
    const auto parent = Parent(id);
    return ot::TraceId(parent.trace_id);
  }
  static talon::TalonObjectReader Reader(const std::string& address) {
    auto client = talon::TalonClient::Make(address, 8388608, 1).ValueOrDie();
    return client->OpenObject("aws", "bucket", "key", std::nullopt).ValueOrDie();
  }
  void SetUp() override { ASSERT_TRUE(SetTracerProvider(MakeProvider(data_)).ok()); }
  void TearDown() override {
    EXPECT_TRUE(SetTracerProvider(nullptr).ok());
    EXPECT_TRUE(SetTraceOptions({}).ok());
  }
  std::shared_ptr<Memory> data_;
};

TEST_F(TalonTracingTest, SdkAndWireUseTheStorageProviderAndActualParent) {
  // W3C parsing in Talon must not turn an in-process Storage parent into a
  // remote parent and select a different host sampling policy.
  ASSERT_TRUE(SetTracerProvider(MakeProvider(data_, true)).ok());
  StatPeer peer;
  auto reader = Reader(peer.address());
  {
    auto parent = AttachParent(Parent());
    TraceScope operation("storage.read");
    auto result = reader.StatAsync();
    ASSERT_TRUE(peer.Wait());
    peer.Release();
    ASSERT_TRUE(result.status().ok()) << result.status();
    EXPECT_EQ(*result.result(), 7);
  }
  auto spans = data_->GetSpans();
  ASSERT_GE(spans.size(), 3);
  const sdk::SpanData *storage = nullptr, *stat = nullptr, *rpc = nullptr;
  for (const auto& span : spans) {
    EXPECT_EQ(span->GetTraceId(), TraceId());
    if (span->GetName() == "storage.read")
      storage = span.get();
    if (span->GetName() == "talon.stat")
      stat = span.get();
    if (span->GetName() == "talon.rpc")
      rpc = span.get();
  }
  ASSERT_NE(storage, nullptr);
  ASSERT_NE(stat, nullptr);
  ASSERT_NE(rpc, nullptr);
  EXPECT_EQ(stat->GetParentSpanId(), storage->GetSpanId());
  EXPECT_EQ(rpc->GetParentSpanId(), stat->GetSpanId());
  EXPECT_EQ(rpc->GetSpanKind(), ot::SpanKind::kClient);
  char id[16];
  rpc->GetSpanId().ToLowerBase16(id);
  EXPECT_EQ(peer.version, 2);
  ASSERT_EQ(peer.parent.size(), 55);
  EXPECT_EQ(peer.parent.substr(36, 16), std::string(id, sizeof(id)));
  EXPECT_EQ(peer.state, "vendor=value");
}

TEST_F(TalonTracingTest, ProviderRemovalKeepsInflightSnapshotAndDisablesNewRequests) {
  StatPeer old_peer;
  auto old_reader = Reader(old_peer.address());
  auto parent = AttachParent(Parent());
  auto pending = old_reader.StatAsync();
  ASSERT_TRUE(old_peer.Wait());
  ASSERT_TRUE(SetTracerProvider(nullptr).ok());
  StatPeer disabled_peer;
  auto disabled_reader = Reader(disabled_peer.address());
  auto disabled = disabled_reader.StatAsync();
  ASSERT_TRUE(disabled_peer.Wait());
  disabled_peer.Release();
  ASSERT_TRUE(disabled.status().ok());
  EXPECT_EQ(disabled_peer.version, 1);
  EXPECT_TRUE(disabled_peer.parent.empty());
  EXPECT_TRUE(data_->GetSpans().empty());
  old_peer.Release();
  ASSERT_TRUE(pending.status().ok());
  EXPECT_GE(data_->GetSpans().size(), 2);
}

TEST_F(TalonTracingTest, UnsampledParentPropagatesWithoutRecording) {
  StatPeer peer;
  auto reader = Reader(peer.address());
  auto parent = AttachParent(Parent(3, false));
  auto result = reader.StatAsync();
  ASSERT_TRUE(peer.Wait());
  peer.Release();
  ASSERT_TRUE(result.status().ok());
  EXPECT_EQ(peer.version, 2);
  ASSERT_EQ(peer.parent.size(), 55);
  EXPECT_EQ(peer.parent.substr(53), "00");
  EXPECT_EQ(peer.state, "vendor=value");
  EXPECT_TRUE(data_->GetSpans().empty());
}

TEST_F(TalonTracingTest, MissingParentDoesNotCreateTalonRootTraces) {
  StatPeer peer;
  auto reader = Reader(peer.address());
  auto result = reader.StatAsync();
  ASSERT_TRUE(peer.Wait());
  peer.Release();
  ASSERT_TRUE(result.status().ok());
  EXPECT_EQ(peer.version, 1);
  EXPECT_TRUE(data_->GetSpans().empty());
}

TEST_F(TalonTracingTest, ReadErrorsRetainCallerBufferAndReachProvider) {
  auto client = talon::TalonClient::Make("127.0.0.1:0", 8388608, 1).ValueOrDie();
  auto reader = client->OpenObject("aws", "bucket", "key", talon::TalonObjectStat{7, "v1"}).ValueOrDie();
  auto parent = AttachParent(Parent());
  uint8_t destination = 42;
  auto result = reader.ReadAtAsync(0, 1, &destination);
  EXPECT_FALSE(result.status().ok());
  EXPECT_EQ(destination, 42);
  auto spans = data_->GetSpans();
  bool found = false;
  for (const auto& span : spans) {
    if (span->GetName() == "talon.read") {
      found = true;
      EXPECT_EQ(span->GetStatus(), ot::StatusCode::kError);
      EXPECT_EQ(span->GetTraceId(), TraceId());
    }
  }
  EXPECT_TRUE(found);
}
TEST_F(TalonTracingTest, SharedReaderUsesEachCallsParent) {
  auto client = talon::TalonClient::Make("127.0.0.1:0", 8388608, 1).ValueOrDie();
  auto reader = client->OpenObject("aws", "bucket", "key", talon::TalonObjectStat{7, "v1"}).ValueOrDie();
  for (uint8_t id : {4, 5}) {
    auto parent = AttachParent(Parent(id));
    uint8_t destination = 0;
    EXPECT_FALSE(reader.ReadAtAsync(0, 1, &destination).status().ok());
    auto spans = data_->GetSpans();
    ASSERT_FALSE(spans.empty());
    for (const auto& span : spans) EXPECT_EQ(span->GetTraceId(), TraceId(id));
  }
}
TEST_F(TalonTracingTest, SpanBudgetPreservesWireParentWhenSdkSpansAreSuppressed) {
  ASSERT_TRUE(SetTraceOptions({true, 1}).ok());
  StatPeer peer;
  auto reader = Reader(peer.address());
  {
    auto parent = AttachParent(Parent());
    TraceScope operation("storage.read");
    auto result = reader.StatAsync();
    ASSERT_TRUE(peer.Wait());
    peer.Release();
    ASSERT_TRUE(result.status().ok());
  }
  auto spans = data_->GetSpans();
  ASSERT_EQ(spans.size(), 1);
  char id[16];
  spans[0]->GetSpanId().ToLowerBase16(id);
  ASSERT_EQ(peer.parent.size(), 55);
  EXPECT_EQ(peer.parent.substr(36, 16), std::string(id, sizeof(id)));
  EXPECT_GT(opentelemetry::nostd::get<int64_t>(spans[0]->GetAttributes().at("storage.spans.dropped")), 0);
}
}  // namespace
}  // namespace milvus_storage::tracing
