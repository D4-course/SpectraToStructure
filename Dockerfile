FROM continuumio/anaconda3

WORKDIR /src/
COPY ./requirements/conda_requirements.yml requirements.yml
COPY ./requirements/conda_pip_req.yml ./pip_ones.yml

RUN conda env create -n main --file ./requirements.yml

RUN . /root/.bashrc && \
    conda init bash &&  \
    conda activate main
RUN apt-get update && apt-get install -y gcc

COPY ./requirements/conda_pip_for_pip.txt pip2.txt

RUN echo "conda activate main" >> ~/.bashrc
RUN conda run -n main pip install -r ./pip2.txt

COPY ./source ./source
COPY ./trainedModels ./trainedModels
COPY ./data ./data
#CMD ["/bin/bash"]
#RUN conda env update --name main --file ./pip_ones.yml
#
#
#CMD ["bash"]