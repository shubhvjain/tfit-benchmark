FROM --platform=linux/amd64 bioconductor/bioconductor_docker:RELEASE_3_16

# Install R reticulate
RUN R -e "install.packages('reticulate', repos='https://cloud.r-project.org')"

# Install Python deps via pip (faster, no conda solver issues)
RUN python3 -m pip install --no-cache-dir \
    pandas \
    python-dotenv \
    pathlib \
    && rm -rf /root/.cache/pip/*

# Your CoRegNet
RUN R -e "BiocManager::install('CoRegNet', update = FALSE, ask = FALSE)"

# Set reticulate to system Python
RUN echo "Sys.setenv(RETICULATE_PYTHON = '/usr/bin/python3')" >> /usr/local/lib/R/etc/Rprofile.site
