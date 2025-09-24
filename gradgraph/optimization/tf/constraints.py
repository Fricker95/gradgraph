#!/usr/bin/env python3
# 
# constraints.py
# 
# Created by Nicolas Fricker on 08/22/2025.
# Copyright © 2025 Nicolas Fricker. All rights reserved.
# 

import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class InBetween(tf.keras.constraints.Constraint):
    """
    Constrains the weights to be within a specified range.

    This constraint clips the weights to lie within the specified lower and upper bounds.

    Parameters
    ----------
    lower_bound : float or None, optional
        The lower bound for the weights. If None, defaults to the smallest positive
        representable number for the current floating-point type (`tf.keras.backend.epsilon()`).
    upper_bound : float or None, optional
        The upper bound for the weights. If None, defaults to the largest representable
        number for the current floating-point type.

    Methods
    -------
    __call__(w)
        Clips the weights `w` to be within the specified bounds.
    get_config()
        Returns the configuration of the constraint as a dictionary.
    from_config(config)
        Instantiates a constraint from a configuration dictionary.

    Examples
    --------
    >>> constraint = InBetween(lower_bound=0.0, upper_bound=1.0)
    >>> model.add(Dense(64, kernel_constraint=constraint))

    Notes
    -----
    This constraint is useful when you want to ensure that the weights of a layer
    remain within a certain range during training.
    """

    def __init__(self, lower_bound: float | None = None, upper_bound: float | None = None) -> None:
        """
        Initialize the class with optional lower and upper bounds.

        Parameters
        ----------
        lower_bound : float or None, optional
            The lower bound for the range. If None, defaults to the smallest positive
            representable number greater than zero, as defined by `tf.keras.backend.epsilon()`.
        upper_bound : float or None, optional
            The upper bound for the range. If None, defaults to the maximum representable
            number for the default float type, as defined by `tf.as_dtype(tf.keras.backend.floatx()).max`.

        Notes
        -----
        This initializer sets the lower and upper bounds for a range, using TensorFlow's
        backend functions to determine default values when bounds are not provided.
        """
        super().__init__()
        self.lower_bound = lower_bound if lower_bound is not None else tf.keras.backend.epsilon()
        self.upper_bound = upper_bound if upper_bound is not None else tf.as_dtype(tf.keras.backend.floatx()).max

    def __call__(self, w):
        """Clip the input tensor values to a specified range.

        Parameters
        ----------
        w : tf.Tensor
            The input tensor to be clipped. The tensor's values will be limited
            to the range defined by `lower_bound` and `upper_bound`.

        Returns
        -------
        tf.Tensor
            A tensor with the same shape and dtype as `w`, where each element
            is clipped to the range [`lower_bound`, `upper_bound`].

        Notes
        -----
        This method casts the `lower_bound` and `upper_bound` attributes to the
        dtype of the input tensor `w` before applying the clipping operation.

        Examples
        --------
        >>> import tensorflow as tf
        >>> class Clipper:
        ...     def __init__(self, lower_bound, upper_bound):
        ...         self.lower_bound = lower_bound
        ...         self.upper_bound = upper_bound
        ...     def __call__(self, w):
        ...         lower_bound = tf.cast(self.lower_bound, w.dtype)
        ...         upper_bound = tf.cast(self.upper_bound, w.dtype)
        ...         return tf.clip_by_value(w, lower_bound, upper_bound)
        >>> clipper = Clipper(0.0, 1.0)
        >>> x = tf.constant([-1.0, 0.5, 2.0])
        >>> clipper(x)
        <tf.Tensor: shape=(3,), dtype=float32, numpy=array([0. , 0.5, 1. ], dtype=float32)>
        """
        lower_bound = tf.cast(self.lower_bound, w.dtype)
        upper_bound = tf.cast(self.upper_bound, w.dtype)
        return tf.clip_by_value(w, lower_bound, upper_bound)

    def get_config(self):
        """Get the configuration of the bounds.

        Returns
        -------
        dict
            A dictionary containing the configuration of the bounds with the following keys:
            - 'lower_bound': The lower bound value.
            - 'upper_bound': The upper bound value.
        """
        return {"lower_bound": self.lower_bound, "upper_bound": self.upper_bound}
