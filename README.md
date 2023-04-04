##  Async Training
The goal here is to compute the average value of all the client.
Each client has a different compute speed than the other clients, depending on the 'compute time' distribution (Normal, Exp, Uniform).
This toy example simulates an asynchronous (federated) learning system.
With some extensions, this should be able to run as a simulation on a GPU.

Run `exp.ipynb` for results

Author: Bart Cox 13-02-2023


### Dev notes

*Using gradients vs model weights*
In async learning, gradients are often used when updating the central server.
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
|Telerig | No     |


### Build

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
