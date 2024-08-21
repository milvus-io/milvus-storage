FROM ubuntu:20.04

RUN apt-get update \
  && apt install -y --no-install-recommends ca-certificates wget curl git g++ gcc make ccache gdb clang-format-10 clang-tidy-10 \
  python3 python3-pip \
  && apt remove --purge -y \
  && rm -rf /var/lib/apt/lists/*

RUN wget -qO- "https://cmake.org/files/v3.27/cmake-3.27.5-linux-`uname -m`.tar.gz" | tar --strip-components=1 -xz -C /usr/local

RUN pip3 install conan==1.61.0

WORKDIR /milvus-storage

COPY . .

RUN rm -r build && python3.8 -m pip install --no-cache-dir -r requirements.txt

CMD ["tail", "-f", "/dev/null"]