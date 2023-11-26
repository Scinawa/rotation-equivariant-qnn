'''
               Max West
    https://arxiv.org/abs/2311.05873
'''

import os, sys
os.environ["KMP_WARNINGS"] = "FALSE" 

from utils import *

import pennylane as qml
import matplotlib.pyplot as plt
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer
import multiprocessing as mp
from functools import partial
from time import time

np.set_printoptions(precision=5, linewidth=150)
np.random.seed(959)

batch_per_it     = 4
batch_size       = 16
num_processes    = 3
rearrange        = 0
num_layers       = int(sys.argv[1])
equivariant      = int(sys.argv[2])
dataset          = sys.argv[3]
id               = int(sys.argv[4])
num_classes      = 10
num_epochs       = 8
num_test         = 500
nqubits          = 13
nfqubits         = 3
opt              = AdamOptimizer(stepsize=1e-3)
loss_history     = []
test_acc_history = []

x_train, y_train, x_test, y_test = load_data(dataset, rearrange, num_classes, num_test)

num_iterations = x_train.shape[0] // (batch_size * batch_per_it)

def fourier():

    for i in range(nfqubits):
        qml.Hadamard(i)
        for j in range(i+1, nfqubits):
            qml.ControlledPhaseShift(2*np.pi/(2**(1+j-i)), wires=[j,i])
        

fourier_wires = { i: nqubits - i - 1  for i in range(nfqubits) }

weights = np.random.randn(num_layers, nqubits - nfqubits * equivariant, 3, requires_grad=True)
print("weights shape: ", weights.shape)

dev = qml.device("default.qubit", wires=nqubits)
@qml.qnode(dev)
def circuit(weights, x):
    
    qml.QubitStateVector(x, wires=range(nqubits))
        
    if equivariant:
        qml.map_wires(qml.adjoint(fourier), fourier_wires)()

    for w in weights:
        for i in range(nqubits - nfqubits * equivariant):
            qml.Rot(*w[i], wires=i)
       
        for i in range(nqubits-1):
            qml.CZ(wires=[i,i+1])
    
    return [ qml.expval(qml.PauliZ(i)) for i in range(num_classes) ]

def acc(weights, data, labels, num_processes, i, return_preds=0):
    
    box     = int(len(data)/num_processes)
    X       = data[i*box:(i+1)*box if (i+2)*box<len(data) else len(data)] 
    labels  = labels[i*box:(i+1)*box if (i+2)*box<len(data) else len(data)]
    
    output  = np.array(circuit(weights, X), requires_grad=False).transpose()
    preds   = np.argmax(output, axis=1) 
    correct = len(labels[labels==preds])
    
    if return_preds:
        return correct, output
    return correct

def cost(weights, X, Y, circuit, precomputed=None):
    
    if precomputed is not None:
        output = precomputed
    else:
        output = 50 * np.array(circuit(weights, X)).transpose()
        Y = Y._value

    log_probs = output - np.log(np.sum(np.exp(output), axis=1,keepdims=True))
    loss      = -np.sum(log_probs[np.arange(Y.shape[0]),Y.astype(int)]) / Y.shape[0]
    
    return loss

if __name__ == '__main__':
    
    pool = mp.Pool(processes=num_processes)
    
    for epoch in range(num_epochs):
        j = 0
        for it in range(num_iterations):
            t = time()   
                
            for i in range(batch_per_it):
                
                batch_index = np.array(range(batch_size*j,batch_size*(j+1)))
                X_train_batch = x_train[batch_index]
                Y_train_batch = y_train[batch_index]
                weights, _, _, _ = opt.step(cost, weights, X_train_batch, Y_train_batch, circuit)
                j += 1

            train_time = time() - t
            t = time()
            
            multi_test   = partial(acc, weights, [ x for x in x_test ],  y_test,  num_processes, return_preds=1)
            test_results = pool.map(multi_test,  [ i for i in range(num_processes) ])
            
            test_acc_history.append(sum([x[0] for x in test_results]) / len(y_test))
            loss_history.append(cost(None, None, y_test, None, precomputed=np.vstack(tuple([x[1] for x in test_results]))).numpy())
            
            if max(test_acc_history) == test_acc_history[-1]:
                np.save(dataset + "_" + str(num_layers) + "_weights_best" + "_equivariant" * equivariant + "_nclasses" + str(num_classes) + "_id" + str(id), weights)

            print( "Epoch: {:3d} | Iter: {:4d} | Test Acc: {:0.5f} | Test Loss: {:0.5f}"
            	"".format(epoch, it + 1, test_acc_history[-1], loss_history[-1]), flush=True)
            
            print("Train time, test time: ", round(train_time, 3), round(time()-t, 3), flush=True)
    
    pool.close()
    pool.join()

    np.save(dataset + "_" + str(num_layers) + "_weights_final" + "_equivariant" * equivariant + "_nclasses" + str(num_classes) + "_id" + str(id), weights)
    print("test_acc", test_acc_history)
