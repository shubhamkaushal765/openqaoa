FROM ubuntu:latest

# Installing python3.10, git and make
RUN apt-get update && \
    apt-get install -y software-properties-common git make

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y python3.10

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --config python3

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --config python

# OpenQAOA workspace
WORKDIR /workspace
RUN git clone -b add-cirq-backend https://github.com/shubhamkaushal765/openqaoa.git
WORKDIR /workspace/openqaoa

CMD ["/bin/bash"]



# docker build -t openqaoa-cirq .
# docker run -it openqaoa-cirq 