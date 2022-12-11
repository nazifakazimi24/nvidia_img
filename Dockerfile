FROM python:3.10

RUN apt-get update \
  && pip3 install --upgrade pip

#WORKDIR /src


RUN mkdir app

WORKDIR ./app

COPY test.py .

#ENV ROKINDATA=/src
CMD ["python", "./test.py"]
