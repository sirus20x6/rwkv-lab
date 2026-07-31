#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>

#include <grpcpp/grpcpp.h>

#include "trainvm/journal.hpp"
#include "trainvm/v1/trainvm.grpc.pb.h"

namespace trainvm {

class AuthorityLock {
 public:
  explicit AuthorityLock(const std::filesystem::path& journal_path);
  ~AuthorityLock();

  AuthorityLock(const AuthorityLock&) = delete;
  AuthorityLock& operator=(const AuthorityLock&) = delete;

 private:
  int descriptor_{-1};
  int journal_descriptor_{-1};
};

class TrainVMService final : public v1::TrainVM::Service,
                             public v1::WorkerControl::Service {
 public:
  explicit TrainVMService(
      const std::filesystem::path& journal_path,
      std::function<std::int64_t()> authority_clock = {});
  ~TrainVMService() override;

  grpc::Status SubmitExperiment(grpc::ServerContext* context,
                                const v1::SubmitExperimentRequest* request,
                                v1::SubmitExperimentResponse* response) override;

  grpc::Status CommandRun(grpc::ServerContext* context,
                          const v1::RunCommandRequest* request,
                          v1::RunCommandResponse* response) override;

  grpc::Status Connect(
      grpc::ServerContext* context,
      grpc::ServerReaderWriter<v1::ControllerToWorker,
                               v1::WorkerToController>* stream) override;

 private:
  struct WorkerConnection {
    WorkerSessionIdentity identity;
    Dispatch dispatch;
    v1::WorkerWelcome welcome;
    std::optional<v1::WorkerReceipt> completed_receipt;
  };

  grpc::Status open_worker_connection(const v1::WorkerHello& hello,
                                      WorkerConnection& connection);
  grpc::Status complete_worker_connection(
      const v1::EventEnvelope& envelope,
      const WorkerConnection& connection, v1::WorkerReceipt& receipt);
  bool claim_worker_attempt(const std::string& key);
  void release_worker_attempt(const std::string& key);
  [[nodiscard]] std::int64_t authority_now_ns() const;

  std::unique_ptr<AuthorityLock> authority_lock_;
  Journal journal_;
  std::mutex command_mutex_;
  std::function<std::int64_t()> authority_clock_;
  std::mutex worker_sessions_mutex_;
  std::set<std::string> active_worker_attempts_;
};

// Blocks until SIGINT or SIGTERM after binding a permission-restricted Unix
// socket. Throws if another authority owns the journal lock.
int serve(const std::filesystem::path& journal_path,
          const std::filesystem::path& socket_path);

}  // namespace trainvm
