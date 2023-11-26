'''
               Max West
    https://arxiv.org/abs/2311.05873
'''

from pennylane import numpy as np

def load_data(dataset, rearrange, num_classes, num_test):

    if "shallow_stm_rot" in dataset:
        
        num_channels = 1
        
        x, y = np.load("./shallow_stm_rot_x.npy"), np.load("./shallow_stm_rot_y.npy")
        x, y = x[y<num_classes], y[y<num_classes]
        x_train, x_test = x[:-num_test], x[-num_test:]
        y_train, y_test = y[:-num_test], y[-num_test:]
        print('x_train.shape: ', x_train.shape, 'y_train.shape: ', y_train.shape, 'x_test.shape: ', x_test.shape, 'y_test.shape: ', y_test.shape, flush=True)
        
        if "radial" in dataset:
            x_train, x_test = x_train[:, np.arange(0, x_train.shape[1], 8)], x_test[:, np.arange(0, x_test.shape[1], 8)]

        print('x_train.shape: ', x_train.shape, 'y_train.shape: ', y_train.shape, 'x_test.shape: ', x_test.shape, 'y_test.shape: ', y_test.shape, flush=True)
        
        return (x_train/np.linalg.norm(x_train, axis=1, keepdims=1)).astype(np.complex128), y_train, (x_test/np.linalg.norm(x_test, axis=1, keepdims=1)).astype(np.complex128), y_test
    
    
    
    if "mnist_rot" in dataset:
        
        num_channels = 1
        print('x_train.shape: ', x_train.shape, 'y_train.shape: ', y_train.shape, 'x_test.shape: ', x_test.shape, 'y_test.shape: ', y_test.shape, flush=True)
        
        return (x_train/np.linalg.norm(x_train, axis=1, keepdims=1)).astype(np.complex128), y_train, (x_test/np.linalg.norm(x_test, axis=1, keepdims=1)).astype(np.complex128), y_test

    if dataset == "fmnist":
        
        num_channels = 1

        x_train = x_train.astype(np.float32)[:,0,:,:]
        x_test  = x_test.astype(np.float32)[:,0,:,:]
        
        x_train = np.pad(x_train, pad_width=((0, 0), (2, 2), (2, 2)), mode='constant', constant_values=0.0)
        x_test = np.pad(x_test, pad_width=((0, 0), (2, 2), (2, 2)), mode='constant', constant_values=0.0)
        
        x_train = x_train.reshape((x_train.shape[0], -1))
        x_test  = x_test.reshape((x_test.shape[0], -1))
    
    if dataset == "cifar10":
    
        num_channels = 3
        
        x_train = np.transpose(x_train, (0,3,1,2))
        x_test  = np.transpose(x_test, (0,3,1,2))  
    
        x_train = x_train.reshape((x_train.shape[0], -1))
        x_test  = x_test.reshape((x_test.shape[0], -1))
        
        x_train = np.pad(x_train, ((0, 0), (512, 512)))
        x_test  = np.pad(x_test,  ((0, 0), (512, 512)))
    
    def get_subset(x, y):

        y = np.argmax(y, axis=1)
        subset = np.where(y < num_classes)
        x = x[subset]
        y = y[subset]
        x /= np.linalg.norm(x, axis=1, keepdims=True)

        return x, y

    x_train, y_train = get_subset(x_train, y_train)
    x_test,  y_test  = get_subset(x_test, y_test)
    
    x_test = x_test[:num_test]
    y_test = y_test[:num_test]
    
    if rearrange:
    
        _map = np.argsort(permute(np.arange(x_train.shape[1]), 32, num_channels))
        
        tmp_train = np.zeros_like(x_train, requires_grad=False)
        tmp_test  = np.zeros_like(x_test,  requires_grad=False)
        
        tmp_train[:,:] = x_train[:,_map]
        tmp_test[:,:]  = x_test[:,_map]
        
        x_train = tmp_train
        x_test  = tmp_test

    print('x_train.shape: ', x_train.shape, 'y_train.shape: ', y_train.shape, 'x_test.shape: ', x_test.shape, 'y_test.shape: ', y_test.shape, flush=True)

    return x_train, y_train, x_test, y_test

def calculate_octagon(l, theta, coords):

    r = l / np.sin(np.pi/4)
    angles = np.linspace(0, 2*np.pi, 9)

    x, y = r * np.cos(angles), r * np.sin(angles)

    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    points = np.vstack((x, y))
    x_rotated, y_rotated = np.dot(rotation_matrix, points)
    coords += [[a,b] for a, b in zip(x_rotated, y_rotated)][:-1]
    
    return coords

def get_coords(abs, angle):

    coords = []
    for i in abs_vals:
        coords = calculate_octagon(i,i*np.pi/angle, coords)

    coords = np.array(coords)
    coords[:,0] -= np.min(coords[:,0])
    coords[:,1] -= np.min(coords[:,1])
    
    return coords

def embed_ims(ims, coords, rotate=0):

    coords = coords.astype(int)
    rotated_coords = np.array([coords[8*(i//8)+((i+rotate)%8)] for i in range(len(coords))])
    print(ims.shape)
    data = ims[:, rotated_coords[:,0], rotated_coords[:,1]]
    
    embedding = np.zeros_like(ims) 
    embedding[:, coords[:,0], coords[:,1]] = data
    
    return data, embedding

def rotate_encoded_data(data, rotations):
    
    # rotations 1d np array e.g. [ 3, 7, 1, 2, 3, 0, ... ]

    assert data.shape[0] == rotations.shape[0]

    rotated_data = np.zeros_like(data)

    for i in range(data.shape[0]):
        rotated_data[i] = data[i][np.array([ 8*(j//8)+((j+rotations[i])%8) for j in range(data.shape[1])])]

    return rotated_data

def data_to_im(data, im_size, coords):
    
    embedding = np.zeros((data.shape[0], im_size, im_size)) - (.4) #0.4
    coords = coords.astype(int)

    embedding[:, coords[:,0], coords[:,1]] = data

    return embedding


#@vectorize([int64(int64, int64, int64)], target='parallel')
#def permute(i,n,c):
#    
#    # maps i to the index related by reflection (see Figure 3 of https://arxiv.org/abs/2212.00264)
#    # where the image size is (c,n,n)
#    # the three channel version is a bit hacky on account of the retrospectively awkward zero padding
#    
#    if c == 1:
#        if (i%n) < n/2:
#            result = int(n*(i//n)/2 + (i%n))
#        else:
#            result = int(n**2 - (n*(i//n)/2 + n-(i%n)) )
#
#    if c == 3:
#
#        if i >= n**2/2 and i < 3*n**2 + n**2/2:
#
#            if (i - n**2/2) % n < n/2:
#                result = int(n**2/2 + ((i - n**2/2)//n)*n/2 + ((i - n**2/2)%n))
#            else:
#                result = int(3*n**2 + n**2/2 - (n*((i - n**2/2)//n)/2 + n - ((i - n**2/2)%n)))
#
#        else:
#            result = i
#
#    return result

#n = 8
#perm = permute(np.arange(n**2), n, 1)
#print('\n')
#for i in range(n**2):
#    print(f'{perm[i]: <4}', end='')
#    if not (i+1)%n: print('', flush=True)
#
#def cost(weights, X, Y, circuit, batch_size):
#    
#    output = 5 * np.array(circuit(weights, X)).transpose()
#    log_probs = output - np.log(np.sum(np.exp(output), axis=1,keepdims=True))
#    
#    return -np.sum(log_probs[np.arange(batch_size),Y._value.astype(int)]) / batch_size 
#
