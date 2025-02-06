FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel
RUN pip install --upgrade peft==0.14.0 safetensors==0.4.5 datasets==3.2.0 accelerate==1.2.1 protobuf==5.29.1 sentencepiece==0.2.0 bitsandbytes==0.45.0 alpaca-eval==0.6.6
