# FROM cvlization-torch-gpu
# FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

RUN pip install wandb==0.16.* torchvision==0.17.1 lightning==2.2.1
RUN pip install einops==0.7.* moviepy==1.0.*
RUN pip install diffusers==0.26.3 timm==0.9.16
RUN pip install accelerate==0.28.*
RUN pip install xformers==0.0.25.post1  # needed by ColossalAI open-sora (STDiT denoising model)
RUN pip install blobfile==1.0.* 

# ADD https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.4.tar.bz2 .
# RUN tar xf openmpi-3.1.4.tar.bz2 \
#     && cd openmpi-3.1.4 \
#     && ./configure --prefix=$MPI_DIR \
#     && make -j4 all \
#     && make install \
#     && cd .. && rm -rf \
#     openmpi-3.1.4 openmpi-3.1.4.tar.bz2 /tmp/*

# RUN groupadd -r mpitest \
#     && useradd -r -g mpitest $USER \
#     && chown -R mpitest:mpitest $HOME

# RUN pip install mpi4py==3.1.*