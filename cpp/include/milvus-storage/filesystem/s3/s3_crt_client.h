// Copyright 2024 Zilliz
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

#ifdef WITH_CRT

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include <arrow/result.h>

#include <aws/s3-crt/S3CrtClient.h>
#include <aws/s3-crt/S3CrtClientConfiguration.h>

#include "milvus-storage/filesystem/s3/s3_client_builder.h"

namespace milvus_storage {

class S3CrtClientHolder;
class RequestCredentialsResolver;

template <>
struct ClientBuilderTraits<Aws::S3Crt::S3CrtClient> {
  using ConfigType = Aws::S3Crt::S3CrtClientConfiguration;
  using HolderType = S3CrtClientHolder;
};

template <>
arrow::Result<std::shared_ptr<S3CrtClientHolder>> ClientBuilder<Aws::S3Crt::S3CrtClient>::BuildClient(
    std::optional<arrow::io::IOContext> io_context, std::shared_ptr<FilesystemMetrics> metrics);

class S3CrtClientOperationState;
class S3CrtClientFinalizer;

/// CRT client ownership and shutdown model
/// ---------------------------------------
///
/// Motivation: why this cannot reuse the regular S3Client lifetime model
/// ----------------------------------------------------------------------
///
/// Aws::S3::S3Client has no CRT-native release-and-wait stage in its
/// destructor:
///
///   S3Client::~S3Client()
///     -> ShutdownSdkClient()
///     -> return
///
/// Aws::S3Crt::S3CrtClient owns an additional asynchronous native client. Its
/// destructor releases that native client and then waits for the native
/// shutdown callback to signal completion:
///
///   S3CrtClient::~S3CrtClient()
///     -> aws_s3_client_release()
///     -> wait on m_clientShutdownSem
///          ^
///          | native shutdown completes only after outstanding meta requests
///          | and their native callbacks have finished
///
/// Consequently, allowing an operation guard to share ownership of the CRT
/// client can make the operation thread decide where the client is destroyed.
/// If a native callback releases the last owner, the resulting wait is cyclic:
///
///   native callback
///     -> release last client owner
///     -> S3CrtClient destructor waits for native shutdown
///     -> native shutdown waits for this callback to return
///
/// The regular S3ClientLock model permits an operation guard to own a client
/// shared_ptr. This CRT-specific model instead requires that:
///
///   * the holder remains the sole shared owner of the CRT client;
///   * operations borrow the client without owning it;
///   * holder finalization waits for all borrowed operations to finish;
///   * every CRT client destructor returns before Aws::ShutdownAPI() begins.
///
/// The construction, operation, and live-client counters below form that
/// shutdown barrier. They close the gaps between building a client, registering
/// it, draining its operations, and completing its native destructor.
///
/// Each holder is the sole shared owner of one S3CrtClient. Operations borrow
/// the client through move-only leases; a lease shares only the small
/// per-holder operation state and never shares ownership of the client or its
/// holder.
///
///   S3CrtClientFinalizer
///            ^       |
///     shared |       | weak registry
///            |       v
///   S3CrtClientHolder -------- sole shared owner ------> S3CrtClient
///            |                                                  ^
///            | shared                                           | raw borrow
///            v                                                  |
///   S3CrtClientOperationState <--------- shared -------- S3CrtClientLease
///
/// The finalizer-to-holder edge is weak while the holder-to-finalizer edge is
/// strong. This avoids an ownership cycle, yet keeps the global finalizer alive
/// until every registered client has reported the completion of its destructor.
///
/// A holder follows this state machine:
///
///   OPEN
///     |  Acquire(): active_operations++
///     |  Lease release: active_operations--
///     |
///     +-- Finalize() --> CLOSING
///                           | reject every new Acquire()
///                           | wait until active_operations == 0
///                           v
///                      destroy client
///                           |
///                           v
///                    report ClientDestroyed()
///
/// S3CrtClientHolder::Finalize() may be entered by global S3 shutdown or by
/// holder destruction.
/// The last holder reference must therefore be released on a thread from which
/// waiting for outstanding native callbacks is safe, never from one of those
/// callbacks itself.

/// Per-holder synchronization block shared by the holder and all outstanding
/// leases. Keeping this state separate from the holder lets a lease decrement
/// the active-operation count after holder finalization has started without
/// owning the holder or the CRT client. The definition remains in the .cpp
/// because callers only need to carry it indirectly through S3CrtClientLease.

/// A move-only lease for one CRT client operation.
///
/// The lease deliberately contains only a non-owning client pointer and
/// operation state. In particular, it must never own S3CrtClient,
/// S3CrtClientHolder, or ObjectCrtInputFile. This makes it safe to release the
/// lease on a CRT callback thread without running the CRT client destructor
/// there. Acquire() increments the shared active-operation count; destroying or
/// overwriting the lease decrements it exactly once and wakes a waiting holder
/// when the count reaches zero.
class S3CrtClientLease {
  public:
  S3CrtClientLease() = default;
  S3CrtClientLease(const S3CrtClientLease&) = delete;
  S3CrtClientLease& operator=(const S3CrtClientLease&) = delete;
  S3CrtClientLease(S3CrtClientLease&& other) noexcept;
  S3CrtClientLease& operator=(S3CrtClientLease&& other) noexcept;
  ~S3CrtClientLease();

  Aws::S3Crt::S3CrtClient* operator->() const;

  protected:
  friend class S3CrtClientHolder;
  void Release();

  Aws::S3Crt::S3CrtClient* client_ = nullptr;
  // Acquire() copies this shared_ptr from the owning holder. The two
  // shared_ptr instances refer to the same per-holder operation-state object,
  // allowing this lease to decrement and notify the state observed by Holder.
  std::shared_ptr<S3CrtClientOperationState> operation_state_;
};

/// Owns one CRT client and gates its destruction on outstanding leases.
///
/// The holder never lends shared ownership of client_. A raw client pointer is
/// valid through a lease because Finalize() cannot move or destroy client_
/// before that lease has decremented active_operations.
class S3CrtClientHolder {
  public:
  S3CrtClientHolder(const S3CrtClientHolder&) = delete;
  S3CrtClientHolder& operator=(const S3CrtClientHolder&) = delete;
  S3CrtClientHolder(S3CrtClientHolder&&) = delete;
  S3CrtClientHolder& operator=(S3CrtClientHolder&&) = delete;

  /// Acquire a non-owning client pointer protected by an operation lease.
  /// No lock is held while the caller uses the client.
  arrow::Result<S3CrtClientLease> Acquire();
  /// The last holder reference must not be released from a native CRT callback.
  ~S3CrtClientHolder();
  std::shared_ptr<FilesystemMetrics> GetMetrics() const;

  protected:
  friend class S3CrtClientFinalizer;
  S3CrtClientHolder(std::shared_ptr<S3CrtClientFinalizer> finalizer,
                    std::shared_ptr<FilesystemMetrics> metrics,
                    std::shared_ptr<RequestCredentialsResolver> credentials_resolver = nullptr);
  void Finalize();

  std::shared_ptr<S3CrtClientFinalizer> finalizer_;
  // Created once for this holder and copied into every successful lease.
  // Holder and all outstanding leases therefore coordinate through the same
  // operation-state object, not through separate counters.
  std::shared_ptr<S3CrtClientOperationState> operation_state_;
  // The holder is the sole shared owner of the client. Operation leases must
  // never copy this shared_ptr.
  std::shared_ptr<Aws::S3Crt::S3CrtClient> client_;
  std::shared_ptr<FilesystemMetrics> metrics_;
  std::shared_ptr<RequestCredentialsResolver> credentials_resolver_;
};

/// Coordinates construction and destruction of every CRT client in the process.
///
/// Shutdown has three phases:
///
///   1. Set finalized_ and wait for every reserved construction to finish.
///   2. Promote registered weak holders, then finalize them without mutex_ held.
///   3. Wait for live_clients_ to reach zero, including holder destructors that
///      started before their weak registry entries were collected.
///
///   AddClient                                  Finalize
///   ---------                                  --------
///   reserve constructing_clients_              finalized_ = true
///   run factory without mutex_ held             wait constructing_clients_ == 0
///   register holder and client                  collect live weak holders
///   increment live_clients_                     holder->Finalize() without mutex_
///   release construction reservation           wait live_clients_ == 0
///
/// Concurrent Finalize() calls are allowed: only the first caller consumes the
/// weak registry, while every caller waits for the same construction and live
/// client counters to drain.
class S3CrtClientFinalizer : public std::enable_shared_from_this<S3CrtClientFinalizer> {
  using ClientHolderList = std::vector<std::weak_ptr<S3CrtClientHolder>>;

  public:
  using ClientFactory = std::function<arrow::Result<std::shared_ptr<Aws::S3Crt::S3CrtClient>>()>;

  /// Reserve construction before invoking the factory, then register its
  /// client. The factory is not invoked after finalization starts.
  arrow::Result<std::shared_ptr<S3CrtClientHolder>> AddClient(
      ClientFactory make_client,
      std::shared_ptr<FilesystemMetrics> metrics,
      std::shared_ptr<RequestCredentialsResolver> credentials_resolver = nullptr);
  /// Wait for client construction, then close all holders and wait until every
  /// CRT client destructor has returned.
  /// This is an S3 lifecycle operation and must not be called from a CRT
  /// callback.
  void Finalize();

  protected:
  friend class S3CrtClientConstructionLease;
  friend class S3CrtClientHolder;
  void ClientDestroyed();

  // Protects holders_ and both counters. It is never held while invoking a
  // client factory, waiting on a holder, or running an S3CrtClient destructor.
  std::mutex mutex_;
  // Wakes finalizers when either construction reservations or live clients
  // reach zero. Both predicates are always checked while mutex_ is held.
  std::condition_variable cv_;
  // Weak ownership is intentional: ordinary filesystem ownership decides when
  // a holder dies; the registry is used only to find holders during shutdown.
  ClientHolderList holders_;
  // Reservations acquired before running a client factory. The RAII
  // construction lease decrements this counter on every failure path.
  std::size_t constructing_clients_ = 0;
  // Clients installed into holders whose destructors have not yet returned.
  // This is stronger than counting lockable weak holders because a weak entry
  // may expire while its holder is still inside its destructor.
  std::size_t live_clients_ = 0;
  // Global admission gate read by Acquire() without taking mutex_. Counter and
  // registry synchronization still uses mutex_.
  std::atomic<bool> finalized_{false};
};

std::shared_ptr<S3CrtClientFinalizer> GetCrtClientFinalizer();

}  // namespace milvus_storage

#endif  // WITH_CRT
