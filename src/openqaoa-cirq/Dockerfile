FROM ubuntu:latest

# Installing python3.10, git and make
RUN apt-get update && \
    apt-get install -y software-properties-common git make python3-pip

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y python3.10

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --config python3

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --config python

# OpenQAOA workspace
WORKDIR /workspace
# RUN git clone -b add-cirq-backend https://github.com/shubhamkaushal765/openqaoa.git
COPY clone.sh /workspace
RUN chmod +x clone.sh
RUN /workspace/clone.sh
WORKDIR /workspace/openqaoa

# Install dependencies for docplex: the default setuptools package
# was unable to find the install_layout option during installation
RUN pip install --upgrade setuptools

RUN make dev-install
RUN make dev-install-tests
RUN make dev-install-docs
RUN make dev-install-all

RUN pip install jupyter-console
RUN python -m ipykernel install --user --name env


CMD ["/bin/bash"]


# docker build -t openqaoa-cirq . | docker run --rm -it openqaoa-cirq
# docker run --rm -it  openqaoa-cirq 
