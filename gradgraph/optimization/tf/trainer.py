#!/usr/bin/env python3
# 
# trainer.py
# 
# Created by Nicolas Fricker on 08/22/2025.
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

    def compile(self, local_optimizer, global_optimizers, *args, **kwargs):
        if isinstance(global_optimizers, dict):
            self.global_optimizers = global_optimizers
            for name, opt in global_optimizers.items():
                setattr(self, f'{name}_global_optimizer', opt)
        else:
            raise ValueError(f"global_optimizers must be a dict")
        super().compile(optimizer=local_optimizer, *args, **kwargs)

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
            config['global_optimizers'] = self.global_optimizers
            return config

    def compile_from_config(self, config):
        config = tf.keras.utils.deserialize_keras_object(config)
        if "optimizer" in config:
            config["local_optimizer"] = config.pop("optimizer")
        self.compile(**config)
        if self.built:
            if hasattr(self, "optimizer") and hasattr(self.optimizer, "build"):
                self.optimizer.build(self.local_trainable_weights)
            if hasattr(self, "global_optimizers"):
                for name, optimizer in self.global_optimizers.items():
                    if not hasattr(optimizer, "build"):
                        continue
                    weights = [w for w in self.global_trainable_weights if w.name == name]
                    if not weights:
                        continue
                    optimizer.build(weights)

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

            pred = y_pred
            loss = self.compute_loss(
                x=x,
                y=y,
                y_pred=pred,
                sample_weight=sample_weight
            )
            self._loss_tracker.update_state(
                unscale_loss_for_distribution(loss),
                sample_weight=tf.shape(
                    next(i for i in tf.keras.tree.flatten(x) if i is not None)
                )[0]
            )
            if self.optimizer is not None:
                loss = self.optimizer.scale_loss(loss)
            if self.global_optimizers is not None:
                g_losses = {k: o.scale_loss(loss) for k, o in self.global_optimizers.items()}

        logs = self.compute_metrics(x, y, pred, sample_weight=sample_weight)

        # Compute Local gradients
        if self.local_trainable_weights:
            trainable_weights = self.local_trainable_weights
            gradients = tape.gradient(loss, trainable_weights)
            local_grad_norm = tf.linalg.global_norm(gradients)
            self.optimizer.apply_gradients(zip(gradients, trainable_weights))

            logs = logs | {'local_grad_norm': local_grad_norm}

        else:
            warnings.warn("The model does not have any local trainable weights.")

        # Compute Global gradients
        if len(self.global_trainable_weights):
            for trainable_weight in self.global_trainable_weights:
                name = trainable_weight.name
                optimizer = self.global_optimizers.get(name, None)
                if optimizer is None:
                    warnings.warn(f"{name} does not have its own optimizer")
                    continue
                loss = g_losses.get(name, None)
                if loss is None:
                    warnings.warn(f"{name} does not have a loss")
                    continue
                gradients = tape.gradient(loss, [trainable_weight])
                global_grad_norm = tf.linalg.global_norm(gradients)
                optimizer.apply_gradients(zip(gradients, [trainable_weight]))
                logs = logs | {name: trainable_weight}
                logs = logs | {f'{name}_global_grad_norm': global_grad_norm}

        else:
            warnings.warn("The model does not have any global trainable weights.")

        del tape

        return logs


