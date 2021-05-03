# Dockerfile for pytorch 
# using pip to install python package
# using multi-stage docker builds
# Been tested on SUrge cluster✔
ARG PYTHON_VERSION=3.6.1
ARG BUILDER_IMG=ubuntu:18.04
ARG RUNTIME_IMG=nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04

FROM ${BUILDER_IMG} AS build-base
LABEL maintainer="suzoosuagr" \
      description="Runtime pytorch cuda image"\
      vendor="SYR-SVS-LAB"\
      version="1.0"

# Passing args.
ARG PYTHON_VERSION
ARG BUILDER_IMG
ARG RUNTIME_IMG

ENV DEBIAN_FRONTEND noninteractive
# install the pyenv build env
RUN apt update \
    && apt install -y --no-install-recommends \
        ca-certificates\
        git\
        wget\
        unzip\
        make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev\
    && apt clean \
    && rm -rf /var/lib/apt/lists/*\
    && git config --global http.sslverify false

FROM build-base AS python
ARG PYTHON_VERSION
ARG BUILDER_IMG
ARG RUNTIME_IMG

# config pyenv and install python
ENV PYENV_ROOT="/pyenv"\
    PATH="/pyenv/shims:/pyenv/bin:$PATH"\
    PYENV_SHELL="bash"
RUN git clone https://github.com/pyenv/pyenv.git /pyenv\
    && cd /pyenv && src/configure && make -C src\
    && pyenv install ${PYTHON_VERSION}\
    && pyenv global ${PYTHON_VERSION}

COPY ./requirements.txt /pyenv
RUN pip install --upgrade --no-cache-dir pip
RUN pip install --upgrade --no-cache-dir -r /pyenv/requirements.txt

FROM ${RUNTIME_IMG} as runtime
ENV PYENV_ROOT="/pyenv"\
    PATH="/pyenv/shims:/pyenv/bin:$PATH"\
    PYENV_SHELL="bash"
COPY --from=python /pyenv /pyenv

WORKDIR /workplace
CMD /bin/bash