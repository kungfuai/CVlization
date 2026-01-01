FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# Pin numpy<2 for package compatibility
RUN pip install "numpy<2"
RUN pip install lightning-bolts==0.7.0 matplotlib==3.9.4 scipy==1.13.1 scikit-learn==1.5.2 scprep==1.2.3 scanpy==1.10.4 torchdyn==1.0.6 pot==0.9.5 torchdiffeq==0.2.5 absl-py==2.1.0 clean-fid==0.1.35
RUN pip install wandb==0.18.3