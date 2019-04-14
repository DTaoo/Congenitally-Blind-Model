import numpy as np
from keras.datasets import mnist
import os
from scipy import misc

def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)  # normalize as it does in DEC paper
    print('MNIST samples', x.shape)
    return x, y


def load_coil_20(resize=(64,64)):
    
    data_dir = 'data/coil-20-proc/'
    data_seq = os.listdir(data_dir)
    data_seq.sort()

    imgs = []
    labels = []

    count = 0
    label = 0

    for i in range(len(data_seq)):
        current_img = misc.imread(data_dir+data_seq[i], mode='L')
        current_img = misc.imresize(current_img, resize)
        imgs.append(np.expand_dims(current_img, axis=3))
        labels.append(label)
        count = count + 1
        if count == 72:
            count = 0
            label = label + 1

     # to satisfiy batchsize of 100
    imgs.extend(imgs[-61:-1])
    labels.extend(labels[-61:-1])
    imgs = np.asarray(imgs)
    labels = np.asarray(labels)
     
    return imgs, labels


def load_coil_10(resize=(64,64)):
    
    data_dir = 'data/coil-10/'
    

    data_seq = os.listdir(data_dir+'train/')
    data_seq.sort()
    x_train = []
    y_train = []
    count = 0
    label = 0
    for i in range(len(data_seq)):
        current_img = misc.imread(data_dir+'train/'+data_seq[i], mode='L')
        current_img = misc.imresize(current_img, resize)
        x_train.append(np.expand_dims(current_img, axis=3))
        y_train.append(label)
        count = count + 1
        if count == 70:
            count = 0
            label = label + 1
    
    data_seq = os.listdir(data_dir+'test/')
    data_seq.sort()
    x_test = []
    y_test = []
    count = 0
    label = 0
    for i in range(len(data_seq)):
        current_img = misc.imread(data_dir+'test/'+data_seq[i], mode='L')
        current_img = misc.imresize(current_img, resize)
        x_test.append(np.expand_dims(current_img, axis=3))
        y_test.append(label)
        count = count + 1
        if count == 10:
            count = 0
            label = label + 1

    x_train = np.asarray(x_train)
    x_test  = np.asarray(x_test)
    y_train = np.asarray(y_train)
    y_test  = np.asarray(y_test)

    return (x_train, y_train), (x_test, y_test)

