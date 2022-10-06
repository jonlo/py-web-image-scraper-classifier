FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
	python3 \
	python3-pip \
	python3-setuptools \
	python3-wheel \
	python3-dev \
	&& rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip
