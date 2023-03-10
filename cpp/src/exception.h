#pragma once

#include <exception>
#include <string>
class StorageException : public std::exception {
 public:
  explicit StorageException(const char *msg) : msg_(msg) {}
  explicit StorageException(const std::string &s) : msg_(s.c_str()) {}
  const char *what() const noexcept override { return msg_; }

 private:
  const char *msg_;
};
