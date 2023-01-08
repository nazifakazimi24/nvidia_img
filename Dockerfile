FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive


RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip


RUN mkdir imgCap_project
WORKDIR /imgCap_project


COPY requirement.txt /imgCap_project/requirement.txt

RUN pip install -r /imgCap_project/requirement.txt


