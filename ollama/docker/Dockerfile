FROM ollama/ollama:0.3.6

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# install python 3.11
RUN apt update && \
    apt install build-essential software-properties-common curl net-tools iputils-ping curl vim git procps -y && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && apt install python3.11 -y && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.11 /usr/bin/python3 && \
    ln -s /usr/bin/python3.11 /usr/bin/python && \
    cd /root && \
    curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py

# install python packages
COPY requirements.txt /root/requirements.txt
RUN pip install --upgrade -r /root/requirements.txt 

