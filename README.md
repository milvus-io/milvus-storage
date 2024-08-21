# Developement Guide

This document is a guide for developers who want to contribute to Milvus Storage. It provides information on the development environment, coding style, and testing process.

## Development Environment

### Prerequisites

- conan >= 1.54.0
- Docker 19.03+
- Docker Compose 1.25+
- Git


### Setup

To set up the development environment, follow these steps:

1. Clone the repository:

```    
git clone https://github.com/milvus-io/milvus-stroage.git
```

2. Build the Milvus Docker image:

```
cd milvus-storage/cpp
docker build -t milvus-storage:latest .
```

3. Start the Docker container:

```
docker run -it milvus-storage:latest bash
```

### Building
```
make build
```

### Testing
```
make test
```

This will run all the tests in the `test` directory.

### Code Style

```
make fix-format
```

This will format the code using `clang-format` and fix any style issues.
