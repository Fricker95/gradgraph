#!/usr/bin/env python3
# 
# trainer.py
# 
# Created by Nicolas Fricker on 08/22/2025.
# Copyright © 2025 Nicolas Fricker. All rights reserved.
# 

import warnings
import tensorflow as tf
from .utils import unscale_loss_for_distribution

class BasePDESystemTrainer(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def local_weights(self):
        return [w for w in self.weights if getattr(w, "_is_local", False)]

    @property
    def local_trainable_weights(self):
        return [w for w in self.trainable_weights if getattr(w, "_is_local", False)]

    @property
    def local_non_trainable_weights(self):
        return [w for w in self.non_trainable_weights if getattr(w, "_is_local", False)]

    @property
    def global_weights(self):
        return [w for w in self.weights if not getattr(w, "_is_local", False)]

    @property
    def global_trainable_weights(self):
        return [w for w in self.trainable_weights if not getattr(w, "_is_local", False)]

    @property
    def global_non_trainable_weights(self):
        return [w for w in self.non_trainable_weights if not getattr(w, "_is_local", False)]

    def compile(self, global_optimizer, local_optimizer, *args, **kwargs):
        """
        Compile the trainer with separate global and local optimizers.

        This stores the ``local_optimizer`` on the instance and forwards the
        ``global_optimizer`` (and any additional arguments) to
        ``tf.keras.Model.compile`` via ``super().compile(...)``.

        Parameters
        ----------
        global_optimizer : tf.keras.optimizers.Optimizer or str
            Optimizer used for the outer/global model updates; passed to
            ``super().compile(optimizer=global_optimizer, *args, **kwargs)``.
        local_optimizer : tf.keras.optimizers.Optimizer or str
            Optimizer used for local/inner updates; stored on ``self.local_optimizer``.
        *args
            Positional arguments forwarded to ``tf.keras.Model.compile``.
        **kwargs
            Keyword arguments forwarded to ``tf.keras.Model.compile`` (e.g.,
            ``loss``, ``metrics``, ``loss_weights``, ``weighted_metrics``,
            ``run_eagerly``).

        Notes
        -----
        - This method sets ``self.local_optimizer = local_optimizer``.
        - All additional arguments are forwarded unchanged to Keras' ``compile``.

        Examples
        --------
        >>> trainer.compile(
        ...     global_optimizer="adam",
        ...     local_optimizer="sgd",
        ...     loss="mse",
        ...     metrics=["mae"],
        ... )
        """
        self.local_optimizer = local_optimizer
        super().compile(optimizer=global_optimizer, *args, **kwargs)

    def build(self, input_shape):
        if self.built:
            return
        # initialize PDE System Layer here
        # call pde_system_layer.build(input_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        # return pde_system_layer.compute_output_shape(input_shape)
        raise NotImplementedError()

    def get_config(self):
        config = super().get_config()
        # update config with the __ini__ arguments
        # config.update({...})
        return config

    def get_compile_config(self):
        if self.compiled and hasattr(self, "_compile_config"):
            config = self._compile_config.serialize()
            config['local_optimizer'] = self.local_optimizer
            return config

    def compile_from_config(self, config):
        config = tf.keras.utils.deserialize_keras_object(config)
        self.compile(**config)
        if self.built:
            if hasattr(self, "local_optimizer") and hasattr(self.local_optimizer, "build"):
                self.local_optimizer.build(self.local_trainable_weights)
            if hasattr(self, "optimizer") and hasattr(self.optimizer, "build"):
                self.optimizer.build(self.global_trainable_weights)

    def call(self, inputs, training=False):
        # call PDE System layer call
        raise NotImplementedError()

    @tf.function
    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)

        # Forward pass
        with tf.GradientTape(persistent=True) as tape:
            if self._call_has_training_arg:
                y_pred = self(x, training=True)
            else:
                y_pred = self(x)

            pred = y_pred[0] if isinstance(y_pred, (list, tuple)) else y_pred
            loss = self.compute_loss(
                x=x,
                y=y,
                y_pred=pred,
                sample_weight=sample_weight
            )
            self._loss_tracker.update_state(
                unscale_loss_for_distribution(loss),
                sample_weight=tf.shape(tf.keras.tree.flatten(x)[0])[0],
            )
            if self.local_optimizer is not None:
                l_loss = self.local_optimizer.scale_loss(loss)
            if self.optimizer is not None:
                loss = self.optimizer.scale_loss(loss)

        logs = self.compute_metrics(x, y, y_pred[0], sample_weight=sample_weight)

        # Compute gradients
        if self.local_trainable_weights:
            trainable_weights = self.local_trainable_weights
            gradients = tape.gradient(l_loss, trainable_weights)
            local_grad_norm = tf.linalg.global_norm(gradients)
            self.local_optimizer.apply_gradients(zip(gradients, trainable_weights))

            logs = logs | {'local_grad_norm': local_grad_norm}

        else:
            warnings.warn("The model does not have any local trainable weights.")

        if len(self.global_trainable_weights):
            trainable_weights = self.global_trainable_weights
            gradients = tape.gradient(loss, trainable_weights)
            global_grad_norm = tf.linalg.global_norm(gradients)
            self.optimizer.apply_gradients(zip(gradients, trainable_weights))

            for v in self.global_trainable_weights:
                logs = logs | {v.name: v}

            logs = logs | {'global_grad_norm': global_grad_norm}
            
        else:
            warnings.warn("The model does not have any global trainable weights.")

        del tape

        return logs


