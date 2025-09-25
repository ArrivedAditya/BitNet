# Bitnet.cpp from Microsoft with NLP and RAG Features
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![version](https://img.shields.io/badge/version-1.0-blue)

## Information
Kernels use in LLM model
- I2_S for x86, x86_64 cpus.
- TL1
- TL2

Model that is supported to use in this branch 
- [Bitnet-b1.58-2B-4T](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T)
- [bitnet_b1_58-large](https://huggingface.co/1bitLLM/bitnet_b1_58-large)
- [Bitnet-b1.58-3B](https://huggingface.co/microsoft/BitNet-b1.58-3B)
- [Llama3-8B-1.58-100B-tokens](https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens)
- [Falcon3 Family](https://huggingface.co/collections/tiiuae/falcon3-67605ae03578be86e4e87026)
- [Falcon-E Family](https://huggingface.co/collections/tiiuae/falcon-edge-series-6804fd13344d6d8a8fa71130)

## Installation

### Requirements
- python>=3.9
- cmake>=3.22
- clang>=18
- conda (highly recommend) e.g., miniforge, miniconda.

#### Operating System Instruction for Fulling Requirments
For Windows users, its best use wsl for this thing and install linux distro like.

For Debian/Ubuntu or LTS Linux distro users
- Packages respostory like ![llvm](https://apt.llvm.org/) 

        `bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"`

### Build from source

1. Clone the repo
```bash
git clone --recursive https://github.com/microsoft/BitNet.git
cd BitNet
```
2. Install the dependencies
```bash
# (Recommended) Create a new conda environment
conda create -n bitnet-cpp python=3.9
conda activate bitnet-cpp

pip install -r requirements.txt
```
3. Build the project
```bash
# Manually download the model and run with local path
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

## Usage
Simple, run run.py 

### Customize
Simple edit run.py

