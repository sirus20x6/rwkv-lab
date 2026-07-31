#pragma once

#include <filesystem>
#include <memory>
#include <mutex>

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

class TrainVMService final : public v1::TrainVM::Service {
 public:
  explicit TrainVMService(const std::filesystem::path& journal_path);
  ~TrainVMService() override;

  grpc::Status SubmitExperiment(grpc::ServerContext* context,
                                const v1::SubmitExperimentRequest* request,
                                v1::SubmitExperimentResponse* response) override;

  grpc::Status CommandRun(grpc::ServerContext* context,
                          const v1::RunCommandRequest* request,
                          v1::RunCommandResponse* response) override;

 private:
  std::unique_ptr<AuthorityLock> authority_lock_;
  Journal journal_;
  std::mutex command_mutex_;
};

// Blocks until SIGINT or SIGTERM after binding a permission-restricted Unix
// socket. Throws if another authority owns the journal lock.
int serve(const std::filesystem::path& journal_path,
          const std::filesystem::path& socket_path);

}  // namespace trainvm
