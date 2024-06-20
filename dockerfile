FROM ubuntu:22.04 as base

COPY . /pe_sem2_final_hw

WORKDIR /pe_sem2_final_hw

EXPOSE 8003

RUN apt-get update &&\
    apt-get install -y python3 python3-pip &&\
    apt-get update &&\
    pip install -r requirements.txt

CMD cd src &&\
    uvicorn main:app --host 0.0.0.0 --port 8002