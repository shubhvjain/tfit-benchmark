FROM bioconductor/bioconductor_docker:RELEASE_3_19

RUN Rscript -e "\
install.packages(c('jsonlite', 'arrow'), repos='https://cloud.r-project.org'); \
BiocManager::install(c('TCGAbiolinks', 'SummarizedExperiment'), ask=FALSE, update=FALSE)"

COPY datasets.json /app/datasets.json
WORKDIR /app