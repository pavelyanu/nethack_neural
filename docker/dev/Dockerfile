FROM docker.io/pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential autoconf libtool pkg-config \
    python3-dev python3-pip python3-numpy git flex bison libbz2-dev

WORKDIR /tmp

COPY requirements.txt /tmp

RUN pip install --no-cache-dir -r requirements.txt
