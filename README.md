# Catalyst

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyPi License](https://badgen.net/pypi/license/pip/)](https://pypi.org/project/pip/)

Catalyst is a lightweight framework to simulate **asynchronous federated learning** systems with heterogeneous client speeds.  
It supports multiple algorithms and is designed for easy extensibility and reproducibility.

---

## Getting Started
The code is tested on Ubuntu 24.04 with Python 3.8

### Installation

Setup a virtual environment
```bash
python38 -m venv venv38
source venv38/bin/activate
```

Install dependencies for GPU or CPU:
Cuda:
```bash
pip install -r requirements_cuda117.txt
```

CPU:
```bash
pip install -r requirements.txt
```


### Submodules

The `data-processing` submodule is *optional* and only needed for dataset preparation. It is *not required* to run asynchronous training simulations.

To update and initialize submodules:

```bash
git submodule update --init --recursive
```
---

## Project Overview

Catalyst simulates asynchronous federated learning where:

- Clients have different compute speeds (Normal, Exponential, or Uniform distributions).
- Client updates arrive at the server asynchronously.
- Updates can use either gradients or model weights.
- Designed to eventually support GPU-based large-scale simulations.

Author: Bart Cox — *13-02-2023*

Email: **b.a.cox@tudelft.nl**

---

## Asynchronous Federated Learning

This framework simulates a federated system where each client operates asynchronously. Clients vary in compute speed according to distributions (e.g., Normal, Exponential, Uniform). 

The goal is to compute the global average across all client updates while handling stragglers and asynchronous behavior robustly. The simulator can be extended for GPU execution and more complex FL scenarios.

---

## Implemented Algorithms

| Algorithm | Supported |
|----------|:---------:|
| AFL      |    ✅     |
| BASGD    |    ✅     |
| Kardam   |    ✅     |
| Catalyst |    ✅     |

---

## Running Experiments

To list all available experiments:

```bash
python -m asyncfl.exps
```

To run a specific experiment:

```bash
python -m asyncfl.exps.<exp_name>
```

For example:

```bash
python -m asyncfl.exps.exp50_mnist
```

To process the results of an experiment:

```bash
python -m asyncfl.exps.<exp_name> -o
```

To follow live logging output:

```bash
tail -f debug.log
```
### Example
```bash
python -m asyncfl.exps.exp50_mnist
python -m asyncfl.proc_exp exp50_mnist -t 750
# Visualize results
python -m asyncfl.proc_exp exp50_mnist -w
```
---

## Reproducibility: Paper Results

Experiment scripts used to generate paper figures and tables:

| Experiment | Purpose |
|:-----------|:--------|
| `exp50_mnist` | Figure 3 & MNIST part of Table 1 |
| `exp51_cifar10` | Figure 4 & CIFAR-10 part of Table 1 |
| `exp52_wikitext2` | Figure 5 & WikiText-2 part of Table 1 |
| `exp53_mnist_scaling_all` | Figure 6 |
| `exp54_mnist_scaling_byz` | Figure 7 |

### Processing Results

Commands to regenerate the exact tables and figures:

#### Table 1
```bash
python -m asyncfl.proc_exp exp50_mnist -t 750
python -m asyncfl.proc_exp exp51_cifar10 -t 7500
python -m asyncfl.proc_exp exp52_wikitext2 -t 750
```

#### Figures
```bash
# Figure 3
python -m asyncfl.proc_exp exp50_mnist -w

# Figure 4
python -m asyncfl.proc_exp exp51_cifar10 -w

# Figure 5
python -m asyncfl.proc_exp exp52_wikitext2 -w

# Figure 6
python -m asyncfl.proc_exp exp53_mnist_scaling_all -c

# Figure 7
python -m asyncfl.proc_exp exp54_mnist_scaling_byz -c
```

---

## Repository Structure

```
asyncfl/
 ├── exps/          # Experiment definitions
 ├── proc_exp.py    # Script for processing experiment results
 ├── core/          # Core simulation logic
 └── utils/         # Utility functions
```

---

## License

See `LICENSE.txt` for more information.


## Contributing

Issues and pull requests are welcome!  
Feel free to open a discussion if you'd like to collaborate or suggest improvements.


## Citation

If you are using Catalyst for your work, please cite our paper with:
```bibtex
@article{cox2024asynchronous,
  title={Asynchronous byzantine federated learning},
  author={Cox, Bart and M{\u{a}}lan, Abele and Chen, Lydia Y and Decouchant, J{\'e}r{\'e}mie},
  journal={arXiv preprint arXiv:2406.01438},
  year={2024}
}
```