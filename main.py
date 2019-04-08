# -*- coding: utf-8 -*-
import numpy as np
import random
import os

x_dim = 256
y_dim = 256

def update_weight(W, M, x_t, y_t, y_hat):
    """ Update weight matrix of W and M.
    Equation (8) in the original paper.
    """
    y_dim = y_t.shape[0]
    y_hat += ((y_t * y_t) * 0.01)
    
    dW = y_t[:,np.newaxis].dot( x_t[np.newaxis,:] - (W.T.dot(y_t[:,np.newaxis])).T)
    #dW = dW / y_hat[:,np.newaxis]
    
    for i in range(y_dim):
        if y_hat[i] > 0:
            W[i] += (dW[i] / y_hat[i])
    
    dM = y_t[:,np.newaxis].dot( y_t[np.newaxis,:] - (M.T.dot(y_t[:,np.newaxis])).T)
    #dM = dM / y_hat[:,np.newaxis]
    
    # Set diagonal elements to zero
    for i in range(M.shape[0]):
        dM[i,i] = 0.0
    
    for i in range(y_dim):
        if y_hat[i] > 0:
            M[i] += (dM[i] / y_hat[i])
            
def adjust_y_i(y_t, x_t, y_index, lambd=200.0):
    """ If the node in y_t is not active previously, adjust it.
    Equation (5) in the original paper.
    """
    y_dim = y_t.shape[0]
    
    # Calculate squree L2 norm of x_t
    x_norm_sq = x_t.dot(x_t)    
    y_t_sqs = y_t * y_t # (3,)
    
    y_t_adj = np.empty_like(y_t)
    
    d_i = x_norm_sq - (np.sum(y_t_sqs) - y_t_sqs[y_index])
    d_i_sq = d_i * d_i
    if d_i_sq <= lambd:
        return 0.0
    else:
        if d_i < 0.0:
            print("d_i was negative: {}".format(d_i))
            d_i = 0.0
    return np.sqrt(d_i)


def save_weights(W, M):
    print("saving weights")
    
    dir_name = "saved"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
    file_path = os.path.join(dir_name, "weight")
    np.savez_compressed(file_path,
                        W=W,
                        M=M)


from dataset import Dataset

dataset = Dataset()
data_size = len(dataset) # 10240
epoch_size = 20

data_indices = list(range(data_size))
random.shuffle(data_indices)

W = np.zeros([y_dim, x_dim]) # (3, 2)
M = np.zeros([y_dim, y_dim]) # (3, 3)

# Active set {i}
active_i_s = []
active_counts = np.zeros([y_dim], dtype=np.int32)

# Iteration count for convergence
iteration = 30

y_hat = np.ones([y_dim]) * 1000.0

T = epoch_size * data_size

for t in range(T):
    # Get one input sample x_t
    data_index = data_indices[t%data_size]
    x_t = dataset[data_index]
    
    # Zero initialze y_t
    y_t = np.zeros(y_dim)
    for i in active_i_s:
        # If 'i' is in the active set {i}
        for it in range(iteration):
            # W[i] = (2,)
            # M[i] = (3,)
            a = W[i].dot(x_t) - M[i].dot(y_t)
            # a is scaler
            y_t_i = max(0, a)
        y_t[i] = y_t_i
    
    for i in range(y_dim):
        if active_counts[i] == 0:
            y_t[i] = adjust_y_i(y_t, x_t, i)
                
    for i in range(y_dim):
        if y_t[i] != 0.0:
            active_counts[i] += 1
            
    for i in range(y_dim):
        if (i not in active_i_s) and active_counts[i] > 0:
            active_i_s.append(i)

    update_weight(W, M, x_t, y_t, y_hat)
    
    print("t={}".format(t))

    if t % 1000 == 0:
        save_weights(W, M)
