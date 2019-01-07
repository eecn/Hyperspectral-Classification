FROM nvidia/cuda:9.1-cudnn7-runtime-ubuntu16.04
MAINTAINER Nicolas Audebert (nicolas.audebert@onera.fr)

RUN apt-get update && apt-get install -y --no-install-recommends \
         #build-essential \
         #cmake \
         #git \
         curl \
         bzip2 \
         #vim \
         ca-certificates \
         #libjpeg-dev \
         #libpng-dev \
         libgl1-mesa-glx &&\ 
     rm -rf /var/lib/apt/lists/*
# (libGL is for matplotlib/seaborn)

WORKDIR /workspace/DeepHyperX/
RUN mkdir -p Datasets
COPY . .
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh 
     #&& \
     #/opt/conda/bin/conda install numpy pyyaml scipy ipython mkl mkl-include && \
     #/opt/conda/bin/conda install -c pytorch magma-cuda90 && \
     #/opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH
#RUN pip install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-linux_x86_64.whl
RUN pip install --no-cache-dir -r requirements.txt
#RUN python main.py --download KSC Botswana PaviaU PaviaC IndianPines

EXPOSE 8097

ENTRYPOINT ["sh", "start.sh"]
