# Catalyst

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyPi license](https://badgen.net/pypi/license/pip/)](https://pypi.org/project/pip/)


## Submodules

To clone this repository use `git clone --recursive git:/<repo-url>.git`

Or to update including submodules: `git submodule update --init --recursive`
## Asynchronous Training

The goal here is to compute the average value of all the client.
Each client has a different compute speed than the other clients, depending on the 'compute time' distribution (Normal, Exp, Uniform).
This toy example simulates an asynchronous (federated) learning system.
With some extensions, this should be able to run as a simulation on a GPU.

~~Run `exp.ipynb` for results~~

Author: Bart Cox 13-02-2023

### Dev notes

*Using gradients vs model weights*
In asynchronous learning, gradients are often used when updating the central server.
This works straightforward for single batch updates.
For multi batch updates, the gradients needs to be accumulated during training.

Using weights is much easier to implement. You do not have to think about the length of training (number of local batches).
You just create a new model out of client weights.

#### Implemented Methods

| Method | Status |
| ------ | :----: |
| AFL    | Yes    |
| BASGD  | Yes    |
| Kardam | Yes    |
|Catalyst | Yes    |

### Build (not required)

```bash
python3 -m pip install --upgrade build
python3 -m build
```

### Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install .
```

### Run

To list the available experiments to run, execute:
```bash
python -m asyncfl.exps
```

Experiments are defines in `python` files in the folder `asyncfl.exps`.
To run an experiment run the following command:

```bash
python -m asyncfl.exps.<exp_name>
```

To process the results run:
```bash
python -m asyncfl.exps.<exp_name> -o
```

To follow the logging output do:

```bash
tail -f debug.log
```

### Reproducibility

Experiment scripts for paper results:

- `exp50_mnist`: Figure 3 & MNIST part of Table 1
- `exp51_cifar10`: Figure 4 & CIFAR-10 part of Table 1
- `exp52_wikitext2`: Figure 5 & WikiText-2 part of Table 1
- `exp53_mnist_scaling_all`: Figure 6
- `exp54_mnist_scaling_byz`: Figure 7

Data table cells and figures:
- Table 1
    - `python -m asyncfl.proc_exp exp50_mnist -t 750`
    - `python -m asyncfl.proc_exp exp51_cifar10 -t 7500`
    - `python -m asyncfl.proc_exp exp52_wikitext2 -t 750`
- Figure 3
    - `python -m asyncfl.proc_exp exp50_mnist -w`
- Figure 4
    - `python -m asyncfl.proc_exp exp51_cifar10 -w`
- Figure 5
    - `python -m asyncfl.proc_exp exp52_wikitext2 -w`
- Figure 6
    - `python -m asyncfl.proc_exp exp53_mnist_scaling_all -c`
- Figure 7
    - `python -m asyncfl.proc_exp exp54_mnist_scaling_byz -c`
