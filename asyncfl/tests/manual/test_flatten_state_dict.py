




import torch
from asyncfl.network import flatten_dict, get_model_by_name, unflatten_dict


if __name__ == '__main__':
    print('Running state_dict tests')
    model_name = 'mnist-cnn'
    # Create a model with random weights
    modelA = get_model_by_name(model_name)
    # Create another model with random weights
    modelB = get_model_by_name(model_name)

    # Flatten ModelA into a single vector
    vecA = flatten_dict(modelA)
    # Change some variables in the vector
    vecA[0] = 0
    vecA[4] = 1

    # Reconstruct the vector into modelB
    unflatten_dict(modelB, vecA)

    # Check if the changed values are correctly copied into modelB
    for params in modelB.state_dict().values():
        assert params[0][0][0][0] == 0
        assert params[0][0][0][4] == 1
        break
    
    print('No errors means that the test was successful')
