FROM ubuntu:22.04 as base

COPY . /mplop_hw_3

WORKDIR /mplop_hw_3

EXPOSE 8003

RUN apt-get update &&\
    apt-get install -y python3 python3-pip &&\
    apt-get update &&\
    pip install -r requirements.txt

CMD uvicorn main:app --host 0.0.0.0 --port 8002