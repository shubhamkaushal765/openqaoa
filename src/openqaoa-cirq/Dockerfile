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
RUN git clone -b add-cirq-backend https://github.com/shubhamkaushal765/openqaoa.git
WORKDIR /workspace/openqaoa

# Install dependencies for docplex: the default setuptools package
# was unable to find the install_layout option during installation
RUN pip install --upgrade setuptools


# RUN-ning make dev-install step-by-step
RUN pip install -e ./src/openqaoa-core
RUN pip install -e ./src/openqaoa-qiskit
RUN pip install -e ./src/openqaoa-braket
RUN pip install -e ./src/openqaoa-azure
# RUN pip install -e .
# RUN pip install -e ./src/openqaoa-cirq

CMD ["/bin/bash"]



# docker build -t openqaoa-cirq .
# docker run -it openqaoa-cirq 


# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# amazon-braket-default-simulator 1.23.2 requires pydantic>2, but you have pydantic 1.10.15 which is incompatible.
# amazon-braket-schemas 1.22.0 requires pydantic>2, but you have pydantic 1.10.15 which is incompatible.

# jupyter-events 0.10.0 requires jsonschema[format-nongpl]>=4.18.0, but you have jsonschema 4.17.3 which is incompatible.  
# jupyterlab 4.2.1 requires httpx>=0.25.0, but you have httpx 0.23.3 which is incompatible.

# jupyterlab-server 2.27.2 requires jsonschema>=4.18.0, but you have jsonschema 4.17.3 which is incompatible.
# referencing 0.35.1 requires attrs>=22.2.0, but you have attrs 21.4.0 which is incompatible.