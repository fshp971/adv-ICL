# "Short-length" Adversarial Training Helps LLMs Defend "Long-length" Jailbreak Attacks: Theoretical and Empirical Evidence

This is the official repository for the preprint ["Short-length Adversarial Training Helps LLMs Defend Long-length Jailbreak Attacks: Theoretical and Empirical Evidence"](https://arxiv.org/abs/2502.04204) by Shaopeng Fu, Liang Ding, and Di Wang.

## Installation

- Python 3.11
- CUDA 11.8
- PyTorch 2.5.1

### Build environment via Anaconda

Download and install [Anaconda3](https://www.anaconda.com/download). Then, run following commands:

```bash
# create & activate conda environment
conda create -n adv-ICL python=3.11
conda activate adv-ICL

# install packages
conda install pytorch=2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install --upgrade peft==0.14.0 safetensors==0.4.5 datasets==3.2.0 accelerate==1.2.1 protobuf==5.29.1 sentencepiece==0.2.0 bitsandbytes==0.45.0

# for AlpacaEval evaluation
pip install alpaca-eval==0.6.6
```

### Build environment via Docker

The docker building file is [./Dockerfile](./Dockerfile). Run following commands, and then the built image is adv-ICL:latest.

```bash
docker pull pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel
docker build --tag 'adv-icl' .
```

**PS:** If you plan to use Docker to run your experiments, don't forget to **mount your default cache folder (e.g., `${HOME}/.cache`) to `/root/.cache` in the Docker container**.

## Quick Start

- [./src](./src) stores all experiment source codes.
- [./configs](./configs) stores all configuration files for experiments.
- [./scripts](./scripts) stores all scripts for running experiments.

Please see the [scripts/README.md](./scripts/README.md) for the tutorial of how to run experiments.

## Citation

```
@article{fu2025short,
  title={"Short-length" Adversarial Training Helps LLMs Defend "Long-length" Jailbreak Attacks: Theoretical and Empirical Evidence},
  author={Shaopeng Fu and Liang Ding and Di Wang},
  journal={arXiv preprint arXiv:2502.04204},
  year={2025}
}
```

## Acknowledgment

- AdvBench dataset: [https://github.com/llm-attacks/llm-attacks](https://github.com/llm-attacks/llm-attacks)
- HarmBench dataset: [https://github.com/centerforaisafety/HarmBench](https://github.com/centerforaisafety/HarmBench)
