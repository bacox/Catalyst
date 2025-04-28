# Catalyst

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyPi License](https://badgen.net/pypi/license/pip/)](https://pypi.org/project/pip/)

Catalyst is a lightweight framework to simulate **asynchronous federated learning** systems with heterogeneous client speeds.  
It supports multiple algorithms and is designed for easy extensibility and reproducibility.

---

## 📦 Cloning the Repository

Make sure to clone the repository **with submodules**:

```bash
git clone --recursive git://<repo-url>.git
```

If you've already cloned it without submodules, initialize and update them using:

```bash
git submodule update --init --recursive
```

---

## 🚀 Project Overview

Catalyst simulates asynchronous federated learning where:

- Clients have different compute speeds (Normal, Exponential, or Uniform distributions).
- Client updates arrive at the server asynchronously.
- Updates can use either **gradients** or **model weights**.
- Designed to eventually support GPU-based large-scale simulations.

Author: **Bart Cox** — *13-02-2023*

---

## ⚙️ Development Notes

### Gradients vs Weights

- **Gradients** are typically used for single-batch updates and need to be **accumulated** over multiple batches.
- **Model weights** are simpler to use, as they abstract away the number of local batches:  
  You can directly use the final trained model weights from each client.

---

## 🧪 Implemented Methods

| Method   | Status |
| -------- | :----: |
| AFL      | ✅ Yes |
| BASGD    | ✅ Yes |
| Kardam   | ✅ Yes |
| Catalyst | ✅ Yes |

---

## 🛠️ Build (Optional)

If you want to build the package manually:

```bash
python3 -m pip install --upgrade build
python3 -m build
```

---

## 📥 Installation

Set up a virtual environment and install the project:

```bash
python3 -m venv venv
source venv/bin/activate
pip install .
```

---

## 🏃 Running Experiments

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

---

## 📊 Reproducibility: Paper Results

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

## 🧹 Repository Structure

```
asyncfl/
 ├── exps/          # Experiment definitions
 ├── proc_exp.py    # Script for processing experiment results
 ├── core/          # Core simulation logic
 └── utils/         # Utility functions
```

---

## 📝 License

See `LICENSE.txt` for more information.


## 🤝 Contributing

Issues and pull requests are welcome!  
Feel free to open a discussion if you'd like to collaborate or suggest improvements.
