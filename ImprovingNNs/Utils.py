import numpy as np

def dictionary_to_vector(params_dict):

    flag = True

    for key in params_dict.keys():

        new_v = np.reshape(params_dict[key], (-1, 1))
        if flag:
            theta = new_v
        else:
            theta = np.concatenate((theta, new_v))
        flag = False
    
    return theta

def vector_to_dictionary(vector, layers):
    L = len(layers)
    parameters = {}
    k = 0

    for l in range(1, L):
        w_dim = layers[l] * layers[l - 1]
        b_dim = layers[l]

        temp_dim = k + w_dim

        parameters["W" + str(l)] = vector[
            k:temp_dim].reshape(layers[l], layers[l - 1])
        parameters["b" + str(l)] = vector[
            temp_dim:temp_dim + b_dim].reshape(b_dim, 1)

        k += w_dim + b_dim

    return parameters


def gradients_to_vector(gradients):
    valid_grads = [key for key in gradients.keys()
                   if not key.startswith("dA")]
    L = len(valid_grads)// 2
    flag = True
    
    for l in range(1, L + 1):
        if flag:
            new_grads = gradients["dW" + str(l)].reshape(-1, 1)
            new_grads = np.concatenate(
                (new_grads, gradients["db" + str(l)].reshape(-1, 1)))
        else:
            new_grads = np.concatenate(
                (new_grads, gradients["dW" + str(l)].reshape(-1, 1)))
            new_grads = np.concatenate(
                (new_grads, gradients["db" + str(l)].reshape(-1, 1)))
        flag = False
        
    return new_grads