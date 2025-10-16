#!/usr/bin/env python3
# 
# utils.py
# 
# Created by Nicolas Fricker on 08/22/2025.
# 

import tensorflow as tf

def initialize_tensorflow(use_gpu:bool=True):
    """
    Initializes TensorFlow to use GPUs or CPUs based on the provided configuration.

    This function configures TensorFlow to utilize available GPUs if `use_gpu` is set to True.
    It attempts to set memory growth for GPUs to prevent TensorFlow from allocating all GPU memory
    at once. If no GPUs are available or if `use_gpu` is set to False, it defaults to using CPUs.

    Parameters
    ----------
    use_gpu : bool, optional
        A flag indicating whether to use GPUs if available. Defaults to True.

    Raises
    ------
    RuntimeError
        If memory growth cannot be set after GPUs have been initialized.

    Notes
    -----
    Memory growth must be set before GPUs have been initialized. This function will print
    the number of physical and logical devices available for the specified configuration.

    Examples
    --------
    Initialize TensorFlow to use GPUs:

    >>> initialize_tensorflow(use_gpu=True)

    Initialize TensorFlow to use CPUs:

    >>> initialize_tensorflow(use_gpu=False)
    """
    if use_gpu:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                # for gpu in gpus:
                #     tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                tf.print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                tf.print(e)
            return
    cpus = tf.config.list_physical_devices('CPU')
    if cpus:
        tf.config.set_visible_devices(cpus[0])
        logical_cpus = tf.config.list_logical_devices('CPU')
        tf.print(len(cpus), "Physical CPUs,", len(logical_cpus), "Logical CPUs")

@tf.function
def unscale_loss_for_distribution(value):
    """
    Unscales the given value by the number of replicas in the strategy.

    This function is intended for use with the TensorFlow backend and
    `tf.distribute` strategies. It adjusts the input value by multiplying it
    with the number of replicas in sync, effectively scaling the loss for
    distributed training.

    Parameters
    ----------
    value : tf.Tensor
        The value to be unscaled, typically a loss value in a distributed
        training setup.

    Returns
    -------
    tf.Tensor
        The unscaled value, adjusted by the number of replicas in the current
        distribution strategy.

    Notes
    -----
    This function is only effective when using TensorFlow's distribution
    strategies. If the number of replicas is 1, the input value is returned
    unchanged.

    Examples
    --------
    >>> import tensorflow as tf
    >>> strategy = tf.distribute.MirroredStrategy()
    >>> with strategy.scope():
    ...     loss = tf.constant(2.0)
    ...     unscaled_loss = unscale_loss_for_distribution(loss)
    ...     print(unscaled_loss)
    4.0  # Assuming 2 replicas in sync
    """
    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    if num_replicas > 1:
        value = tf.keras.ops.multiply(value, tf.keras.ops.cast(num_replicas, value.dtype))
    return value

