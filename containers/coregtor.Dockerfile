FROM condaforge/miniforge3:latest

WORKDIR /app

SHELL ["/bin/bash", "-c"]

# Install packages available in conda-forge via mamba
RUN --mount=type=cache,target=/opt/conda/pkgs \
    mamba create -y -p /opt/core-3.11 \
    python=3.11 \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    jupyter \
    seaborn \
    pyyaml \
    requests \
    tqdm \
    -c conda-forge

ENV PATH="/opt/core-3.11/bin:$PATH"

# now pip requirements
COPY containers/coregtor_requirements.txt /app/requirements.txt 

RUN --mount=type=cache,target=/root/.cache/pip \
    source activate /opt/core-3.11 && \
    pip install --upgrade pip && \
    pip install -r /app/requirements.txt
