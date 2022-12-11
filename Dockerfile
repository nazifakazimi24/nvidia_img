FROM ubuntu:latest

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip


RUN mkdir imgCap_project
WORKDIR /imgCap_project


COPY requirements.txt /imgCap_project/requirements.txt

RUN pip install -r /imgCap_project/requirements.txt

COPY . /imgCap_project


CMD ["python", "./query_processing.py"]
