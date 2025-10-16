#!/usr/bin/env python3
# 
# layers.py
# 
# Created by Nicolas Fricker on 08/22/2025.
# 

from __future__ import annotations

import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class Embedding(tf.keras.layers.Embedding):
    """
    Embedding layer that inherits from `tensorflow.keras.layers.Embedding`.

    This layer maps integer indices to dense vectors of fixed size. It is
    typically used to transform categorical data, such as words, into
    continuous vectors for input into a neural network.

    Parameters
    ----------
    input_dim : int
        Size of the vocabulary, i.e., maximum integer index + 1.
    output_dim : int
        Dimension of the dense embedding.
    *args : tuple
        Additional positional arguments passed to the parent class.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    Methods
    -------
    build(input_shape)
        Creates the layer's weights, ensuring that the embeddings are
        trainable if the layer is set to be trainable.

    Notes
    -----
    This class extends `tensorflow.keras.layers.Embedding` and inherits its
    functionality. The `build` method is overridden to ensure that the
    embeddings' trainability is synchronized with the layer's trainable
    attribute.

    Examples
    --------
    >>> import tensorflow as tf
    >>> embedding_layer = Embedding(input_dim=1000, output_dim=64)
    >>> input_data = tf.constant([[1, 2, 3], [4, 5, 6]])
    >>> output = embedding_layer(input_data)
    >>> output.shape
    TensorShape([2, 3, 64])
    """

    def __init__(self, input_dim, output_dim, *args, **kwargs):
        super().__init__(input_dim, output_dim, *args, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        if hasattr(self, '_embeddings'):
            self._embeddings.trainable = self.trainable
            self._embeddings._is_local = getattr(self, '_is_local', False)
            self._embeddings.name = self.name

class BasePDESystemLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_local_weight(self, name, **kwargs):
        v = self.add_weight(name=name, **kwargs)
        v._is_local = True
        return v

    def add_global_weight(self, name, **kwargs):
        v = self.add_weight(name=name, **kwargs)
        v._is_local = False
        return v

    def add_local_embedding(self, name, **kwargs):
        v = Embedding(name=name, **kwargs)
        v._is_local = True
        return v

    def add_global_embedding(self, name, **kwargs):
        v = Embedding(name=name, **kwargs)
        v._is_local = False
        return v

    @property
    def global_weights(self):
        return [w for w in self.weights if not getattr(w, '_is_local', False)]

    @property
    def global_trainable_weights(self):
        return [w for w in self.trainable_weights if not getattr(w, '_is_local', False)]

    @property
    def global_non_trainable_weights(self):
        return [w for w in self.non_trainable_weights if not getattr(w, '_is_local', False)]

    @property
    def local_weights(self):
        return [w for w in self.weights if getattr(w, '_is_local', False)]

    @property
    def local_trainable_weights(self):
        return [w for w in self.trainable_weights if getattr(w, '_is_local', False)]

    @property
    def local_non_trainable_weights(self):
        return [w for w in self.non_trainable_weights if getattr(w, '_is_local', False)]

    def build(self, input_shape) -> None:
        # if self.built:
        #     return
        # Create weights here using add_(local|global)_(weights|embeddings)
        # Build like this
        # ids = input_shape[0]
        # for v in self.local_weights:
        #     _ = v.build(ids)
        super().build(input_shape)

    def body(self, *args, **kwargs):
        raise NotImplementedError()

    def cond(self, *args, **kwargs):
        raise NotImplementedError()

    def call(self, inputs, training=False):
        # tf.while_loop(self.cond, self.body, loop_vars=..., parallel_iterations=1)
        raise NotImplementedError()

    def compute_output_shape(self, input_shape):
        raise NotImplementedError()

    def get_config(self):
        return super().get_config()
