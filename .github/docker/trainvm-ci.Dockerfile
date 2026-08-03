# Toolchain image for the TrainVM native CI job.
#
# The compiler was never the obstacle: the official gcc:16 image accepts
# -freflection. What blocks a stock image is the dependency pair. Debian trixie
# ships gRPCConfig.cmake but no protobuf-config.cmake, because its protobuf is
# autotools-built, so find_package(Protobuf CONFIG REQUIRED) fails. Installing a
# source-built protobuf beside Debian's gRPC gets past that and then fails at
# CMake generate time on a missing ALIAS target, because the two halves no
# longer agree about their protobuf targets.
#
# The fix is to build gRPC from source with its own bundled submodules, which
# installs abseil, protobuf, re2, c-ares, zlib and gRPC together with mutually
# consistent CMake config packages. One build, one set of targets, no mismatch.
#
# Every version is pinned. A silent upstream bump must not change what CI
# proves, and a moving toolchain turns a red build into an archaeology exercise.
FROM gcc:16.1.0-trixie

# Matches the host this project is developed on (gRPC 1.82.1, protobuf 35.x).
# gRPC's submodule pins its own protobuf, so pinning gRPC pins the pair.
ARG GRPC_VERSION=v1.82.1
# Debian trixie's Go is older than the dashboard module requires.
ARG GO_VERSION=1.26.3
ARG GO_SHA256=2b2cfc7148493da5e73981bffbf3353af381d5f93e789c82c79aff64962eb556
# Debian trixie ships CMake 3.31, whose FindSQLite3 module does not define the
# SQLite3::SQLite3 imported target that trainvm/CMakeLists.txt links against.
# find_package(SQLite3 REQUIRED) still succeeds there, so the failure surfaces
# later as a missing ALIAS target rather than a missing package.
ARG CMAKE_VERSION=4.4.0
ARG CMAKE_SHA256=3864eb649b4466ae126a64bbde1657adad78efbbaa068bf38201de5cf1b5349f
# The journal's auxiliary-path authority refuses to run below SQLite 3.53.3, so
# Debian's package is not merely old, it fails the hostd and ledger suites
# outright. Built from source rather than packaged for that reason.
ARG SQLITE_VERSION=3530300
ARG SQLITE_SHA256=c917d7db16648ec95f714974ace5e5dcf46b7dc70e26600a0a102a3141125db0

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake ninja-build git pkg-config curl \
      libssl-dev libyaml-cpp-dev nlohmann-json3-dev \
      python3 python3-venv python3-pip dbus \
 && rm -rf /var/lib/apt/lists/*

# gRPC plus its bundled protobuf/abseil, installed to /usr/local as one
# consistent set. Static libraries keep the CMake target graph simple and mean
# the built binaries do not depend on this image at runtime.
# Only the submodules the build consumes. --recurse-submodules pulls bloaty,
# a size-analysis tool this build never uses, whose pinned protobuf commit
# cannot be fetched shallowly -- it fails the clone outright.
RUN git clone --depth 1 --branch "${GRPC_VERSION}" \
      https://github.com/grpc/grpc /tmp/grpc \
 && git -C /tmp/grpc submodule update --init --depth 1 --recursive \
      third_party/abseil-cpp \
      third_party/protobuf \
      third_party/re2 \
      third_party/zlib \
      third_party/cares/cares \
      third_party/xxhash \
 && cmake -S /tmp/grpc -B /tmp/grpc/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DBUILD_SHARED_LIBS=OFF \
      -DgRPC_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      -DgRPC_SSL_PROVIDER=package \
      -DgRPC_ZLIB_PROVIDER=module \
      -DgRPC_ABSL_PROVIDER=module \
      -DgRPC_CARES_PROVIDER=module \
      -DgRPC_RE2_PROVIDER=module \
      -DgRPC_PROTOBUF_PROVIDER=module \
      -Dprotobuf_INSTALL=ON \
      -DABSL_ENABLE_INSTALL=ON \
 && cmake --build /tmp/grpc/build --target install \
 && rm -rf /tmp/grpc \
 && ldconfig

# SQLite from source. Debian's package is below the floor the journal's
# auxiliary-path authority enforces, so the hostd and ledger suites fail
# outright against it. Placed after gRPC so a SQLite bump does not invalidate
# the expensive gRPC layer.
RUN curl -fsSL "https://sqlite.org/2026/sqlite-autoconf-${SQLITE_VERSION}.tar.gz" \
      -o /tmp/sqlite.tgz \
 && echo "${SQLITE_SHA256}  /tmp/sqlite.tgz" | sha256sum -c - \
 && tar -C /tmp -xzf /tmp/sqlite.tgz \
 && cd "/tmp/sqlite-autoconf-${SQLITE_VERSION}" \
 && ./configure --prefix=/usr/local >/dev/null \
 && make -j"$(nproc)" >/dev/null \
 && make install >/dev/null \
 && cd / && rm -rf /tmp/sqlite* \
 && ldconfig

# Installed AFTER gRPC on purpose. gRPC 1.82's vendored submodules declare old
# cmake_minimum_required values that CMake 4 rejects, so the gRPC layer above
# builds with Debian's 3.31, and the project build gets 4.x from here. Keeping
# the order also means editing this layer does not invalidate the gRPC layer.
RUN curl -fsSL "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" \
      -o /tmp/cmake.tgz \
 && echo "${CMAKE_SHA256}  /tmp/cmake.tgz" | sha256sum -c - \
 && tar -C /usr/local --strip-components=1 -xzf /tmp/cmake.tgz \
 && rm /tmp/cmake.tgz

# The dashboard module requires a Go newer than Debian packages.
RUN curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" \
      -o /tmp/go.tgz \
 && echo "${GO_SHA256}  /tmp/go.tgz" | sha256sum -c - \
 && tar -C /usr/local -xzf /tmp/go.tgz \
 && rm /tmp/go.tgz
ENV PATH="/opt/venv/bin:/usr/local/go/bin:${PATH}"

# The authority reads /etc/machine-id as its host identity. Container images do
# not ship one, so trainvm_tests aborts before it reaches an assertion. A stable
# per-image id is correct here: CI wants a real identity, just not a shared one.
RUN dbus-uuidgen > /etc/machine-id 2>/dev/null || \
    (head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n' > /etc/machine-id) \
 && test -s /etc/machine-id

# The C++/Python parity tests drive the real trainvm binary against the real
# rwkv_lab package, so they need both installed. CPU torch only: this image
# never sees an accelerator, and the parity contracts do not need one.
COPY pyproject.toml /tmp/pkg/pyproject.toml
COPY src /tmp/pkg/src
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir torch \
      --index-url https://download.pytorch.org/whl/cpu \
 && /opt/venv/bin/pip install --no-cache-dir '/tmp/pkg[trainvm-worker]' \
 && rm -rf /tmp/pkg

# Fail the image build, not a confusing CI run three weeks later, if any piece
# of the contract is missing. A toolchain image that silently lacks a CMake
# config package is worse than no image: the native job would go red for a
# reason that has nothing to do with the change under test.
RUN g++ --version | head -1 \
 && mkdir -p /tmp/sq \
 && echo 'consteval int f(){return 1;} int main(){return f()-1;}' > /tmp/r.cpp \
 && g++ -std=c++26 -freflection /tmp/r.cpp -o /tmp/r && /tmp/r \
 && test -f /usr/local/lib/cmake/protobuf/protobuf-config.cmake \
 && test -f /usr/local/lib/cmake/grpc/gRPCConfig.cmake \
 && test -x /usr/local/bin/protoc \
 && test -x /usr/local/bin/grpc_cpp_plugin \
 && cmake --version | head -1 \
 && printf 'cmake_minimum_required(VERSION 3.25)\nproject(t CXX)\nfind_package(SQLite3 REQUIRED)\nif(NOT TARGET SQLite3::SQLite3)\n  message(FATAL_ERROR "FindSQLite3 does not define SQLite3::SQLite3")\nendif()\n' > /tmp/sq/CMakeLists.txt \
 && cmake -S /tmp/sq -B /tmp/sq/b > /dev/null \
 && /usr/local/bin/sqlite3 --version \
 && test "$(printf '%s\n' "3.53.3" "$(/usr/local/bin/sqlite3 --version | cut -d" " -f1)" | sort -V | head -1)" = "3.53.3" \
 && go version \
 && python3 --version \
 && python3 -c "import torch, rwkv_lab, google.protobuf" \
 && test -s /etc/machine-id \
 && rm -rf /tmp/r /tmp/r.cpp /tmp/sq
