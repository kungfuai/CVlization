FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

# Install git, g++
RUN apt-get update -y && apt-get install -y git build-essential \
    python3-dev wget sudo ninja-build 
    # \
    # && apt-get clean && rm -rf /var/lib/apt/lists/*
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get install -y ffmpeg libsm6 # libxext6

COPY examples/instance_segmentation/seem/requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/MaureenZOU/detectron2-xyz.git
RUN pip install git+https://github.com/arogozhnikov/einops.git
RUN pip install pillow==8.3.1 wandb
RUN pip install nltk transformers pyarrow
# ENV MPI_DIR=/opt/ompi
# ENV PATH="$MPI_DIR/bin:$HOME/.local/bin:$PATH"
# ENV LD_LIBRARY_PATH="$MPI_DIR/lib:$LD_LIBRARY_PATH"

# ADD https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.4.tar.bz2 .
# RUN tar xf openmpi-3.1.4.tar.bz2 \
#     && cd openmpi-3.1.4 \
#     && ./configure --prefix=$MPI_DIR \
#     && make -j4 all \
#     && make install \
#     && cd .. && rm -rf \
#     openmpi-3.1.4 openmpi-3.1.4.tar.bz2 /tmp/*
# RUN pip install mpi4py