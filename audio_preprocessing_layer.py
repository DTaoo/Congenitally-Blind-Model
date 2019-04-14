from __future__ import  division
from keras.engine.topology import Layer, InputSpec
import tensorflow as tf
import keras.backend as K
import numpy as np

class spectrogram(Layer):
    # compute the spectrogram of audio
    def __init__(self, frame_len=1024, frame_step=512, fft_len=1024, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(spectrogram, self).__init__(**kwargs)
        self.frame_len  = 512 #frame_len/2
        self.frame_step = 256 #frame_step/2
        self.fft_len    = fft_len
        #self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        self.audio_len  = input_shape[1]
        super(spectrogram, self).build(input_shape)
        #self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, self.audio_len))
        #self.built = True

    def call(self, inputs, **kwargs):

        # normalization
        signals = inputs/32768.0
        # compute stfts
        stfts = tf.contrib.signal.stft(signals, frame_length=self.frame_len, frame_step=self.frame_step,
                                       fft_length=self.fft_len)

        magnitude_spectrograms = tf.abs(stfts)

        #magnitude_spectrograms = tf.expand_dims(magnitude_spectrograms,3)
        return magnitude_spectrograms

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        fft_unique_bins = self.fft_len // 2 + 1
        length = int(np.ceil((self.audio_len-self.frame_len) / self.frame_step))
        return (input_shape[0], length , fft_unique_bins, 1)

    def get_config(self):
        config = {'frame_len':   self.frame_len,
                  'frame_step':  self.frame_step,
                  'fft_len':     self.fft_len
                  }
        base_config = super(spectrogram, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class logSpectrogram(spectrogram):
    # compute the log-spectrogram of audio
    def __init__(self, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(logSpectrogram, self).__init__(**kwargs)


    def build(self, input_shape):
        assert len(input_shape) == 2
        super(logSpectrogram, self).build(input_shape)

    def call(self, inputs, **kwargs):
        spectrogram = super(logSpectrogram, self).call(inputs)
        log_offset = 1e-6
        log_spectrograms = tf.log(spectrogram + log_offset)

        log_spectrograms = tf.expand_dims(log_spectrograms,3)
        return log_spectrograms

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        fft_unique_bins = self.fft_len // 2 + 1
        length = int(np.ceil((self.audio_len-self.frame_len) / self.frame_step))
        return (input_shape[0], length , fft_unique_bins, 1)

    def get_config(self):
        base_config = super(logSpectrogram, self).get_config()
        return base_config



class logMelSpectrogram(spectrogram):
    # compute the log-mel spectrogram of audio
    def __init__(self, lower_edge_hertz=80.0, upper_edge_hertz= 7600.0, num_mel_bins=64, sample_rate = 44100, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(logMelSpectrogram, self).__init__(**kwargs)
        self.lower_edge_hertz = lower_edge_hertz
        self.upper_edge_hertz = upper_edge_hertz
        self.num_mel_bins     = num_mel_bins
        self.sample_rate = sample_rate


    def build(self, input_shape):
        assert len(input_shape) == 2
        super(logMelSpectrogram, self).build(input_shape)


    def call(self, inputs, **kwargs):
        spectrogram = super(logMelSpectrogram, self).call(inputs)

        num_spectrogram_bins = spectrogram.shape[-1].value

        linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
            self.num_mel_bins, num_spectrogram_bins, self.sample_rate, self.lower_edge_hertz,
            self.upper_edge_hertz)
        mel_spectrograms = tf.tensordot(
            spectrogram, linear_to_mel_weight_matrix, 1)

        mel_spectrograms.set_shape(spectrogram.shape[:-1].concatenate(
            linear_to_mel_weight_matrix.shape[-1:]))

        # perform the logarithmic compression over mel scale of spectrogram
        log_offset = 1e-6
        log_mel_spectrograms = tf.log(mel_spectrograms + log_offset)

        log_mel_spectrograms = tf.expand_dims(log_mel_spectrograms,3)
        return log_mel_spectrograms

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        length = int(np.ceil((self.audio_len-self.frame_len) / self.frame_step))
        return (input_shape[0], length, self.num_mel_bins, 1)

    def get_config(self):
        config = {'lower_edge_hertz':   self.frame_len,
                  'upper_edge_hertz':   self.frame_step,
                  'num_mel_bins':       self.fft_len,
                  'sample_rate':        self.sample_rate
                  }

        base_config = super(logMelSpectrogram, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
