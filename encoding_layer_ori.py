from __future__ import division
from keras.engine.topology import Layer, InputSpec
import keras.backend as K
import os, sys, struct, math
from PIL import Image as pil_image
import numpy as np
from scipy.io.wavfile import write as wav_write
import tensorflow as tf
import copy

FL = 80  # Lowest  frequency (Hz) in soundscape
FH = 7600  # Highest frequency (Hz)
FS = 22050  # Sample  frequency (Hz)
T = 1.05 #1.05  # Image to sound conversion time (s)
D = 1  # Linear|Exponential|tanh=0|1|2 distribution
HIFI = 1  # 8-bit|16-bit=0|1 sound quality
STEREO = 0  # Mono|Stereo=0|1 sound selection
DELAY = 1  # Nodelay|Delay=0|1 model   (STEREO=1)
FADE = 1  # Relative fade No|Yes=0|1  (STEREO=1)
DIFFR = 1  # Diffraction No|Yes=0|1    (STEREO=1)
BSPL = 1  # Rectangular|B-spline=0|1 time window
BW = 0  # 16|2-level=0|1 gray format in P[][]
CAM = 1  # Use OpenCV camera input No|Yes=0|1
VIEW = 1  # Screen view for debugging No|Yes=0|1
CONTRAST = 2  # Contrast enhancement, 0=none
PITCH = 0 # the position of high pitch: top|middle|bottom = 0|1|2 


# Coefficients used in rnd()
ir = 0
ia = 9301
ic = 49297
im = 233280

TwoPi = 6.283185307179586476925287
HIST = (1 + HIFI) * (1 + STEREO)
WHITE = 1.00
BLACK = 0.00

N = 64
M = 64


# k     = 0
b = 0
d = D
ns = 2 * int(0.5 * FS * T)
m = int(ns / N)
sso = 0.0 if HIFI else 128
ssm = 32768.0 if HIFI else 128
scale = 0.5 / math.sqrt(M)
dt = 1.0 / FS
v = 340.0  # v = speed of sound (m/s)
hs = 0.20  # hs = characteristic acoustical size of head (m)


def wi(fp, i):
    b0 = int(i % 256)
    b1 = int((i - b0) / 256)
    fp.write(struct.pack('B', b0 & 0xff))
    fp.write(struct.pack('B', b1 & 0xff))


def wl(fp, l):
    i0 = l % 65536
    i1 = (l - i0) / 65536
    wi(fp, i0)
    wi(fp, i1)


def rnd():
    global ir, ia, ic, im
    ir = (ir * ia + ic) % im
    return ir / (1.0 * im)


class Encoding_layer(Layer):
    """
    # This layer is used for encoding the image into audio segment

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```

    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, img_rows, img_cols, img_channels)`.
    # Output shape
        2D tensor with shape: `(n_samples, audio_length)`.
    """

    def __init__(self, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Encoding_layer, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=4)

    def build(self, input_shape):
        assert len(input_shape) == 4
        self.batch_num = input_shape[0]
        self.img_rows = input_shape[1]
        self.img_cols = input_shape[2]
        self.img_channels = input_shape[3]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, self.img_rows, self.img_cols, self.img_channels))
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        inputs = inputs*255
        #inputs = (inputs+1)*127.5       

        k = 0
        w_pre = [0 for i in range(M)]
        phi0_pre = [0 for i in range(M)]
 
        for i in range(0, M): phi0_pre[i] = TwoPi * rnd()

        # Set lin|exp (0|1) frequency distribution and random initial phase 
        if d==1:
            for i in range(0, M): w_pre[i] = TwoPi * FL * pow(1.0 * FH / FL, 1.0 * i / (M - 1))
        elif d==2:
            fre = FH-FL
            for i in range(0, M): w_pre[i] = TwoPi * FL + TwoPi * (fre / 2 * np.tanh(0.06*(i-M/2)) + fre/2) 
        else:
            for i in range(0, M): w_pre[i] = TwoPi * FL + TwoPi * (FH - FL) * i / (M - 1)


        if PITCH == 0:
            w = w_pre
            phi0 = phi0_pre
        elif PITCH == 1:
            w = w_pre[0:None:2]
            w_half_tail = w_pre[1:None:2]
            w_half_tail.reverse()
            w.extend(w_half_tail)
            
            phi0 = phi0_pre[0:None:2]
            phi0_half_tail = phi0_pre[1:None:2]
            phi0_half_tail.reverse()
            phi0.extend(phi0_half_tail)

        elif PITCH == 2:
            w_pre.reverse()
            w = w_pre
            phi0_pre.reverse()
            phi0 = phi0_pre

        # convert to gray scale,         image scale 0-1 ?  0-255 ?
        if self.img_channels == 3:
            inputs = tf.image.rgb_to_grayscale(inputs)

        imgs = tf.image.resize_nearest_neighbor(inputs, [M, N])
        #imgs = inputs
        imgs_reverse = tf.reverse(imgs, [1])

        # step 1
        avg = tf.reduce_mean(imgs,axis=[1,2,3], keep_dims=True)
        #avg = tf.reshape(avg,[-1, self.img_rows, self.img_cols])

        px = imgs_reverse + CONTRAST*tf.subtract(imgs_reverse, avg)
        px = tf.maximum(px, 0.0)
        px = tf.minimum(px, 255.0)

        base = tf.constant(10.0,dtype='float32')
        zero = tf.constant(0.0,dtype='float32')
        A = tf.where(tf.equal(px, zero), px, tf.pow(base, (px / 16 - 15) / 10.0))


        # step 2
        tau1 = 0.5 / w[M - 1]
        tau2 = 0.25 * (tau1 * tau1)
        y = yl = yr = z = zl = zr = 0.0

        # expanding A
        num = int(np.floor(ns/N))
        B = tf.reshape(A, [self.batch_num, M, N, 1])
        A_expand = tf.reshape(tf.tile(B, [1, 1, 1, num]),[self.batch_num, M, N*num])
        A_expand_revese = tf.slice(A_expand, [0,0,self.img_cols*num-(ns-N*num)], [-1,-1,ns-N*num])
        A_expand = tf.concat([A_expand, A_expand_revese], axis=2)
        A_expand = tf.transpose(A_expand, perm=[0,2,1])

        frames = tf.reshape(tf.tile(tf.range(ns, dtype='float32'), [self.batch_num]), [-1, ns])
        frames = tf.tile(tf.reshape(frames, [self.batch_num, ns, 1]), [1, 1, M])
        frames_dt = frames * dt

        w = np.expand_dims(np.expand_dims(w,axis=0), axis=1)
        phi0 = np.expand_dims(np.expand_dims(phi0,axis=0), axis=1)
        s = tf.reduce_sum(A_expand * tf.sin(w*frames_dt+phi0), axis=-1)

        '''
        yp = y
        y = tau1 / dt + tau2 / (dt * dt)
        y = (s + y * yp + tau2 / dt * z) / (1.0 + y)
        z = (y - yp) / dt
        '''

        # Laplacian kernel
        #s = tf.expand_dims(s,2)
        #filter = tf.constant([1,2,-6,2,1], dtype='float32')
        #filter = tf.expand_dims(tf.expand_dims(filter,1),2)
        #y = tf.nn.conv1d(value=s, filters=filter, stride=1, padding='SAME',data_format='NHWC')
        y = tf.squeeze(s)

        l = sso + 0.5 + scale * ssm * y  # y = 2nd order filtered s
        l = tf.maximum(l, sso - ssm)
        audio = tf.minimum(l, sso - 1 + ssm)

        return audio

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 4
        return (input_shape[0], ns)

    def get_config(self):
        # config = {'audio_len': self.audio_len}
        base_config = super(Encoding_layer, self).get_config()
        # return dict(list(base_config.items()) + list(config.items()))
        return base_config
