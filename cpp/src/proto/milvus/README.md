# Vendored Milvus Plan Protobuf

This directory contains generated C++ protobuf sources copied from the Milvus repository for storage-side predicate delete evaluation.

Source location used for this copy:

```text
/home/hanchun/Documents/project/milvus/internal/core/src/pb/
```

Copied files:

```text
plan.pb.h
plan.pb.cc
schema.pb.h
schema.pb.cc
common.pb.h
common.pb.cc
```

`plan.pb.h` depends on `schema.pb.h` and `common.pb.h`, so these generated files must be kept together and synchronized as one set.

These files are internal implementation dependencies only. Do not expose them from `cpp/include/milvus-storage/`, and do not make them part of the public C/C++ API.

When Milvus `pkg/proto/plan.proto`, `schema.proto`, or `common.proto` changes in a way that affects predicate delete plan serialization, regenerate/copy the matching generated C++ files from the same Milvus commit and update this directory together.
