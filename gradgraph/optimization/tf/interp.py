#!/usr/bin/env python3
# 
# interp.py
# 
# Created by Nicolas Fricker on 08/22/2025.
# 

import tensorflow as tf

def batch_linear_interp_1d(
    x,
    x_ref_min,
    x_ref_max,
    y_ref,
    clamp=True,
    fill_value_below=0.0,
    fill_value_above=0.0,
):
    """
    Perform batched differentiable 1D linear interpolation with shape safety.

    Parameters
    ----------
    x : array_like
        Array of shape `(batch, n_interp)` containing x values to interpolate.
    x_ref_min : array_like or scalar
        Array of shape `(batch,)` or a scalar representing the start of the 
        reference grid.
    x_ref_max : array_like or scalar
        Array of shape `(batch,)` or a scalar representing the end of the 
        reference grid.
    y_ref : array_like
        Array of shape `(batch, nx)` containing y values at regularly spaced 
        grid points.
    clamp : bool, optional
        If True, clamps x to the range `[x_ref_min, x_ref_max]`. Default is True.
    fill_value_below : scalar, optional
        Scalar value to use for out-of-bound handling below `x_ref_min` if 
        `clamp` is False. Default is 0.0.
    fill_value_above : scalar, optional
        Scalar value to use for out-of-bound handling above `x_ref_max` if 
        `clamp` is False. Default is 0.0.

    Returns
    -------
    y_interp : array_like
        Array of shape `(batch, n_interp)` containing the interpolated y values.

    Notes
    -----
    This function assumes that `y_ref` is defined on a regularly spaced grid 
    between `x_ref_min` and `x_ref_max`. The interpolation is performed in a 
    batched manner, allowing for efficient computation over multiple sets of 
    input data.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[0.5, 1.5], [2.0, 3.0]])
    >>> x_ref_min = np.array([0.0, 1.0])
    >>> x_ref_max = np.array([2.0, 3.0])
    >>> y_ref = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
    >>> batch_linear_interp_1d(x, x_ref_min, x_ref_max, y_ref)
    array([[0.5, 1.5],
           [2.0, 3.0]])
    """
    x = tf.convert_to_tensor(x)
    dtype = x.dtype
    y_ref = tf.convert_to_tensor(y_ref, dtype=dtype)

    batch_size = tf.shape(x)[0]
    n_interp = tf.shape(x)[1]
    nx = tf.shape(y_ref)[1]

    x_ref_min = tf.reshape(x_ref_min, [-1, 1])  # (batch, 1)
    x_ref_max = tf.reshape(x_ref_max, [-1, 1])  # (batch, 1)

    if clamp:
        x_clamped = tf.clip_by_value(x, x_ref_min, x_ref_max)
    else:
        x_clamped = x

    t = tf.cast((x_clamped - x_ref_min) / (x_ref_max - x_ref_min + 1e-8), dtype)

    x_idx = t * tf.cast(nx - 1, dtype)

    i0 = tf.floor(x_idx)
    i1 = i0 + 1

    i0 = tf.clip_by_value(i0, 0.0, tf.cast(nx - 1, dtype))
    i1 = tf.clip_by_value(i1, 0.0, tf.cast(nx - 1, dtype))

    i0 = tf.cast(i0, tf.int32)
    i1 = tf.cast(i1, tf.int32)

    w1 = x_idx - tf.cast(i0, dtype)
    w0 = 1.0 - w1

    batch_idx = tf.range(batch_size)[:, tf.newaxis]          # (batch, 1)
    batch_idx = tf.tile(batch_idx, [1, n_interp])            # (batch, n_interp)

    i0 = tf.reshape(i0, [batch_size, n_interp])
    i1 = tf.reshape(i1, [batch_size, n_interp])

    idx0 = tf.stack([batch_idx, i0], axis=-1)  # (batch, n_interp, 2)
    idx1 = tf.stack([batch_idx, i1], axis=-1)

    y0 = tf.gather_nd(y_ref, idx0)
    y1 = tf.gather_nd(y_ref, idx1)

    y_interp = w0 * y0 + w1 * y1

    if not clamp:
        below = tf.less(x, x_ref_min)  # (batch, n_interp)
        above = tf.greater(x, x_ref_max)
        y_interp = tf.where(below, fill_value_below, y_interp)
        y_interp = tf.where(above, fill_value_above, y_interp)

    return y_interp

