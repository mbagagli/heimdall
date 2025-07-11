# HEIMDALL installation

This document shows how to set up the **heimdall_v030** environment after you have cloned the project from GitHub.

-------------------------------------------------------------------------------------------------------------------

First let's clone the repository on you local machine

```bash
git clone https://github.com/mbagagli/heimdall.git
cd heimdall
```

The software is shipped with two YAML files:

* `envs/heimdall_testing_linux.yml`: for Linux/Unix with NVIDIA GPU (CUDA 12.1)
* `envs/heimdall_testing_mac.yml`: for macOS (CPU or Apple-silicon MPS)

If you're intalling on Linux machines, first let's check the NVIDA driver support installation:
```bash
nvidia-smi # should confirm driver >= 550
```
If nothing pops up on screen, you need to install the CUDA support on your machine first.
Please note that HEIMDALL hase been tested on `CUDA version 12.1`, so anythong `12.x` should work too.

Then (either for both Mac or Linux installation):
```bash
conda env create -f envs/heimdall_testing_linux.yaml[envs/heimdall_testing_macOS.yml]
conda activate heimdall_v030
pip install .  # mind the dot !!!
```

To check the installation was OK, run (on the project root):

```bash
pytest tests/test_installation.py
```

If all are passed (ignore the warnings) you're ready to go.
