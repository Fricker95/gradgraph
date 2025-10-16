#!/usr/bin/env python3
# 
# callbacks.py
# 
# Created by Nicolas Fricker on 08/31/2025.
# 

import warnings
import numpy as np
import tensorflow as tf

class EarlyStoppingByThreshold(tf.keras.callbacks.Callback):
    """
    Early stopping callback to terminate training when a monitored metric reaches a specified threshold.

    This callback is used to stop training when a monitored metric reaches a specified threshold, 
    which can be useful to prevent overfitting or to save computational resources.

    Parameters
    ----------
    monitor : str, optional
        The metric to be monitored. Default is 'val_loss'.
    threshold : float, optional
        The threshold value that the monitored metric must reach to stop training. Default is 0.
    min_delta : float, optional
        Minimum change in the monitored metric to qualify as an improvement. Default is 0.
    baseline : float, optional
        Baseline value for the monitored metric. Training will stop if the model does not show improvement over the baseline. Default is None.
    verbose : int, optional
        Verbosity mode. 0 = silent, 1 = progress messages. Default is 0.
    mode : {'auto', 'min', 'max'}, optional
        Mode for determining whether the monitored metric should be minimized or maximized. Default is 'auto'.
    restore_best_weights : bool, optional
        Whether to restore model weights from the epoch with the best monitored metric value. Default is False.
    start_from_epoch : int, optional
        The epoch from which to start monitoring the metric. Default is 0.

    Raises
    ------
    ValueError
        If the mode is not recognized or if the monitored metric cannot be automatically determined to be minimized or maximized.

    Warns
    -----
    UserWarning
        If the mode is unknown, or if the monitored metric is not available in the logs.

    Examples
    --------
    >>> early_stopping = EarlyStoppingByThreshold(monitor='val_accuracy', threshold=0.95, mode='max')
    >>> model.fit(X_train, y_train, callbacks=[early_stopping])
    """

    def __init__(
        self,
        monitor="val_loss",
        threshold=0,
        min_delta=0,
        baseline=None,
        verbose=0,
        mode="auto",
        restore_best_weights=False,
        start_from_epoch=0,
    ) -> None:
        """
        Initialize the EarlyStoppingByThreshold callback.

        This callback stops training once a monitored metric reaches a specified threshold.
        It supports both minimizing and maximizing metrics and can optionally restore the
        model weights from the best epoch.

        Parameters
        ----------
        monitor : str, optional
            Name of the metric to monitor (e.g., 'val_loss', 'val_accuracy').
            Defaults to 'val_loss'.

        threshold : float, optional
            The threshold value that the monitored metric must reach to trigger stopping.
            For 'min' mode, training stops when the metric is less than or equal to this value.
            For 'max' mode, when it is greater than or equal to this value. Defaults to 0.

        min_delta : float, optional
            Minimum change in the monitored metric to qualify as an improvement.
            Used to prevent stopping for negligible changes. Defaults to 0.

        baseline : float or None, optional
            Baseline value for the monitored metric. Training will stop if the model does not
            show improvement over the baseline. Defaults to None.

        verbose : int, optional
            Verbosity mode. 0 = silent, 1 = prints stopping messages. Defaults to 0.

        mode : {'auto', 'min', 'max'}, optional
            Mode for interpreting the monitored metric.
            - 'min': training stops when the monitored metric has stopped decreasing.
            - 'max': training stops when the monitored metric has stopped increasing.
            - 'auto': automatically infers direction from the name of the monitored metric.
            Defaults to 'auto'.

        restore_best_weights : bool, optional
            If True, restores model weights from the epoch with the best monitored value.
            Defaults to False.

        start_from_epoch : int, optional
            Epoch number from which to start monitoring. Defaults to 0.

        Raises
        ------
        ValueError
            If `mode` is not one of {'auto', 'min', 'max'}.

        Warns
        -----
        UserWarning
            If `mode` is unrecognized, it falls back to 'auto'.

        """
        super().__init__()
        self.monitor = monitor
        self.threshold = threshold
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.best_epoch = 0
        self.start_from_epoch = start_from_epoch

        if mode not in ["auto", "min", "max"]:
            warnings.warn(
                f"EarlyStopping mode {mode} is unknown, fallback to auto mode.",
                stacklevel=2,
            )
            mode = "auto"
        self.mode = mode
        self.monitor_op = None

    def _set_monitor_op(self):
        """
        Set the monitoring operation for early stopping based on the specified mode.

        This method determines the appropriate TensorFlow comparison operation
        (`tf.math.less` or `tf.math.greater`) to use for monitoring a specified metric
        during training. The operation is set based on the `mode` attribute or inferred
        from the metric's characteristics.

        Raises
        ------
        ValueError
            If the `monitor` attribute is set to a metric that cannot be automatically
            determined to be maximized or minimized.

        Notes
        -----
        - If `mode` is "min", the monitoring operation is set to `tf.math.less`.
        - If `mode` is "max", the monitoring operation is set to `tf.math.greater`.
        - If the metric name is "loss", the monitoring operation defaults to
        `tf.math.less`.
        - If the metric has a `_direction` attribute, it is used to determine the
        monitoring operation.
        - The `min_delta` attribute is negated if the monitoring operation is
        `tf.math.less`.
        - The `best` attribute is initialized to positive or negative infinity based
        on the monitoring operation.
        """
        if self.mode == "min":
            self.monitor_op = tf.math.less
        elif self.mode == "max":
            self.monitor_op = tf.math.greater
        else:
            metric_name = self.monitor.removeprefix("val_")
            if metric_name == "loss":
                self.monitor_op = tf.math.less
            if hasattr(self.model, "metrics"):
                for m in self.model.metrics:
                    if m.name == metric_name:
                        if hasattr(m, "_direction"):
                            if m._direction == "up":
                                self.monitor_op = tf.math.greater
                            else:
                                self.monitor_op = tf.math.less
        if self.monitor_op is None:
            raise ValueError(
                f"EarlyStopping callback received monitor={self.monitor} "
                "but Keras isn't able to automatically determine whether "
                "that metric should be maximized or minimized. "
                "Pass `mode='max'` in order to do early stopping based "
                "on the highest metric value, or pass `mode='min'` "
                "in order to use the lowest value."
            )
        if self.monitor_op == tf.math.less:
            self.min_delta *= -1
        self.best = (
            float("inf") if self.monitor_op == tf.math.less else -float("inf")
        )

    def on_train_begin(self, logs = None) -> None:
        """
        Executes actions at the beginning of the training process.

        This method is typically called at the start of the training process to 
        initialize or reset any necessary states or variables.

        Parameters
        ----------
        logs : dict, optional
            Currently, this parameter is not used. It is included for compatibility 
            with similar methods that may require logging information.

        Notes
        -----
        This method calls the `_reset` function to ensure that the training state 
        is initialized properly before training begins.
        """
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch: int, logs = None) -> None:
        """
        Callback function to be called at the end of each epoch to adjust the learning rate.

        This function checks the monitored metric and adjusts the learning rate of the optimizer
        if the metric has not improved for a specified number of epochs (`patience`). It also
        handles cooldown periods and ensures the learning rate does not fall below a minimum value.

        Parameters
        ----------
        epoch : int
            The index of the current epoch.
        logs : dict, optional
            A dictionary of logs from the current epoch. If not provided, an empty dictionary
            is used.

        Warns
        -----
        UserWarning
            If the monitored metric is not available in the logs.

        Notes
        -----
        This function assumes that the model has an optimizer attribute and that the optimizer
        has a `learning_rate` attribute. The learning rate is reduced by a factor when the
        monitored metric does not improve for a specified number of epochs.

        Examples
        --------
        >>> # Assuming `self` is an instance of a class with the necessary attributes
        >>> self.on_epoch_end(epoch=5, logs={'accuracy': 0.8})
        """
        if self.monitor_op is None:
            self._set_monitor_op()

        current = self.get_monitor_value(logs)
        if current is None or epoch < self.start_from_epoch:
            return
        if self.restore_best_weights and self.best_weights is None:
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch

        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            if self.monitor_op(current, self.threshold) and epoch > 0:
                self.stopped_epoch = epoch
                self.model.stop_training = True
            return

    def on_train_end(self, logs=None):
        """
        Handles operations to be performed at the end of training.

        This method is typically used in a training loop to manage actions
        such as early stopping and restoring model weights to the best
        observed state during training.

        Parameters
        ----------
        logs : dict, optional
            Currently not used. Defaults to `None`.

        Notes
        -----
        - If early stopping was triggered (i.e., `self.stopped_epoch > 0`),
          and verbosity is enabled (`self.verbose > 0`), a message indicating
          the epoch at which training was stopped is printed.
        - If `self.restore_best_weights` is `True` and `self.best_weights` is
          not `None`, the model's weights are restored to the best observed
          state. A message is printed if verbosity is enabled.

        Examples
        --------
        >>> class Model:
        ...     def __init__(self):
        ...         self.stopped_epoch = 5
        ...         self.verbose = 1
        ...         self.restore_best_weights = True
        ...         self.best_weights = [0.1, 0.2, 0.3]
        ...         self.best_epoch = 3
        ...         self.model = self
        ...     def set_weights(self, weights):
        ...         print("Weights set to:", weights)
        ...     def on_train_end(self, logs=None):
        ...         if self.stopped_epoch > 0 and self.verbose > 0:
        ...             print(f"Epoch {self.stopped_epoch + 1}: early stopping")
        ...         if self.restore_best_weights and self.best_weights is not None:
        ...             if self.verbose > 0:
        ...                 print("Restoring model weights from the end of the best epoch:", f"{self.best_epoch + 1}.")
        ...             self.set_weights(self.best_weights)
        >>> model = Model()
        >>> model.on_train_end()
        Epoch 6: early stopping
        Restoring model weights from the end of the best epoch: 4.
        Weights set to: [0.1, 0.2, 0.3]
        """
        if self.stopped_epoch > 0 and self.verbose > 0:
            tf.print(
                f"Epoch {self.stopped_epoch + 1}: early stopping"
            )
        if self.restore_best_weights and self.best_weights is not None:
            if self.verbose > 0:
                tf.print(
                    "Restoring model weights from "
                    "the end of the best epoch: "
                    f"{self.best_epoch + 1}."
                )
            self.model.set_weights(self.best_weights)

    def get_monitor_value(self, logs):
        """
        Retrieve the value of the monitored metric from the logs.

        Parameters
        ----------
        logs : dict
            A dictionary containing the metrics and their corresponding values. 
            If `None`, an empty dictionary is used.

        Returns
        -------
        monitor_value : any
            The value of the monitored metric specified by `self.monitor`. 
            If the metric is not found in `logs`, `None` is returned.

        Warns
        -----
        UserWarning
            If the monitored metric specified by `self.monitor` is not found in `logs`, 
            a warning is issued indicating the available metrics.

        Examples
        --------
        >>> class Monitor:
        ...     def __init__(self, monitor):
        ...         self.monitor = monitor
        ...     def get_monitor_value(self, logs):
        ...         logs = logs or {}
        ...         monitor_value = logs.get(self.monitor)
        ...         if monitor_value is None:
        ...             warnings.warn(
        ...                 (
        ...                     f"Early stopping conditioned on metric `{self.monitor}` "
        ...                     "which is not available. "
        ...                     f"Available metrics are: {','.join(list(logs.keys()))}"
        ...                 ),
        ...                 stacklevel=2,
        ...             )
        ...         return monitor_value
        >>> monitor = Monitor('accuracy')
        >>> logs = {'loss': 0.25, 'val_loss': 0.3}
        >>> monitor.get_monitor_value(logs)
        UserWarning: Early stopping conditioned on metric `accuracy` which is not available. Available metrics are: loss,val_loss
        None
        """
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                (
                    f"Early stopping conditioned on metric `{self.monitor}` "
                    "which is not available. "
                    f"Available metrics are: {','.join(list(logs.keys()))}"
                ),
                stacklevel=2,
            )
        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        """
        Determine if the monitored value shows improvement over the reference value.

        This method evaluates whether the `monitor_value` indicates an improvement
        when compared to the `reference_value`. The improvement is assessed using
        the `monitor_op` operation, considering a minimum change threshold defined
        by `min_delta`.

        Parameters
        ----------
        monitor_value : float
            The current value of the metric being monitored.
        reference_value : float
            The reference value to compare against, typically the best recorded
            value so far.

        Returns
        -------
        bool
            True if the `monitor_value` shows improvement over the `reference_value`
            according to the `monitor_op` operation and `min_delta` threshold;
            False otherwise.

        Notes
        -----
        The `monitor_op` is a callable that determines the comparison logic, such
        as whether an increase or decrease in the `monitor_value` is considered
        an improvement. The `min_delta` is a small positive number that defines
        the minimum change required to qualify as an improvement.
        """
        return self.monitor_op(monitor_value - self.min_delta, reference_value)

class ReduceLROnPlateau(tf.keras.callbacks.Callback):
    """
    Reduce learning rate when a metric has stopped improving.

    This callback is used to reduce the learning rate by a specified factor
    when a monitored metric has stopped improving. It helps in fine-tuning
    the learning process by decreasing the learning rate when the model
    reaches a plateau in its performance.

    Parameters
    ----------
    optimizer : str, optional
        The optimizer attribute of the model to adjust the learning rate for.
        Default is 'optimizer'.
    monitor : str, optional
        The metric to be monitored. Default is 'val_loss'.
    factor : float, optional
        Factor by which the learning rate will be reduced. `new_lr = lr * factor`.
        Must be less than 1.0. Default is 0.1.
    patience : int, optional
        Number of epochs with no improvement after which learning rate will be reduced.
        Default is 10.
    verbose : int, optional
        Verbosity mode. 0 = silent, 1 = update messages. Default is 0.
    mode : {'auto', 'min', 'max'}, optional
        Mode for reducing the learning rate. In 'min' mode, the learning rate
        will be reduced when the monitored quantity has stopped decreasing; in
        'max' mode it will be reduced when the monitored quantity has stopped
        increasing; in 'auto' mode, the direction is automatically inferred from
        the name of the monitored quantity. Default is 'auto'.
    min_delta : float, optional
        Threshold for measuring the new optimum, to only focus on significant changes.
        Default is 1e-4.
    cooldown : int, optional
        Number of epochs to wait before resuming normal operation after the learning
        rate has been reduced. Default is 0.
    min_lr : float, optional
        Lower bound on the learning rate. Default is 0.0.

    Raises
    ------
    ValueError
        If `factor` is greater than or equal to 1.0.

    Warns
    -----
    UserWarning
        If the `mode` is unknown, it falls back to 'auto' mode.
        If the `monitor` metric is not available in the logs.

    Examples
    --------
    >>> reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    >>> model.fit(x_train, y_train, callbacks=[reduce_lr])
    """

    def __init__(
        self,
        optimizer='optimizer',
        monitor="val_loss",
        factor=0.1,
        patience=10,
        verbose=0,
        mode="auto",
        min_delta=1e-4,
        cooldown=0,
        min_lr=0.0,
        **kwargs,
    ):
        """
        Initialize the ReduceLROnPlateau scheduler.

        Parameters
        ----------
        optimizer : str, optional
            The optimizer to be used. Default is 'optimizer'.
        monitor : str, optional
            The metric to be monitored. Default is 'val_loss'.
        factor : float, optional
            Factor by which the learning rate will be reduced. 
            `new_lr = lr * factor`. Must be less than 1.0. Default is 0.1.
        patience : int, optional
            Number of epochs with no improvement after which learning rate will be reduced. Default is 10.
        verbose : int, optional
            Verbosity mode. Default is 0.
        mode : {'auto', 'min', 'max'}, optional
            One of `{'auto', 'min', 'max'}`. In 'min' mode, the learning rate will be reduced when the quantity monitored has stopped decreasing; in 'max' mode it will be reduced when the quantity monitored has stopped increasing; in 'auto' mode, the direction is automatically inferred from the name of the monitored quantity. Default is 'auto'.
        min_delta : float, optional
            Threshold for measuring the new optimum, to only focus on significant changes. Default is 1e-4.
        cooldown : int, optional
            Number of epochs to wait before resuming normal operation after lr has been reduced. Default is 0.
        min_lr : float, optional
            Lower bound on the learning rate. Default is 0.0.
        **kwargs
            Additional arguments passed to the superclass initializer.

        Raises
        ------
        ValueError
            If `factor` is greater than or equal to 1.0.

        Notes
        -----
        This class is typically used to reduce the learning rate when a metric has stopped improving. The reduction is multiplicative, and the learning rate is reduced by `factor` when no improvement is seen for a `patience` number of epochs.

        Examples
        --------
        >>> scheduler = ReduceLROnPlateau(optimizer='adam', monitor='val_accuracy', factor=0.5, patience=5)
        """
        super().__init__(**kwargs)

        self.optimizer = optimizer
        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError(
                "ReduceLROnPlateau does not support a factor >= 1.0. "
                f"Received factor={factor}"
            )

        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """
        Resets the wait counter and cooldown counter for learning rate adjustment.

        This method reinitializes the internal state used for tracking the
        performance of a monitored metric and adjusts the mode of operation
        if necessary.

        Warns
        -----
        UserWarning
            If the learning rate reduction mode is unknown, a warning is issued
            and the mode is set to 'auto'.

        Notes
        -----
        The method sets the `monitor_op` and `best` attributes based on the
        current mode and monitored metric. If the mode is 'min' or 'auto' with
        a non-accuracy metric, the `monitor_op` is set to detect decreases in
        the monitored value. Otherwise, it is set to detect increases.

        The `cooldown_counter` and `wait` attributes are reset to zero.
        """
        if self.mode not in {"auto", "min", "max"}:
            warnings.warn(
                f"Learning rate reduction mode {self.mode} is unknown, "
                "fallback to auto mode.",
                stacklevel=2,
            )
            self.mode = "auto"
        if self.mode == "min" or (
            self.mode == "auto" and "acc" not in self.monitor
        ):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        """
        Resets the internal state at the beginning of training.

        This method is called at the start of training to initialize or reset
        internal counters such as cooldown and wait, and to prepare for tracking
        the monitored metric.

        Parameters
        ----------
        logs : dict, optional
            Currently unused. Reserved for future use or compatibility with the
            Keras callback API. Defaults to None.
        """
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch to monitor and adjust the learning rate.

        If the monitored metric has not improved for a number of epochs defined
        by `patience`, and the cooldown period has elapsed, the learning rate is
        reduced by the specified `factor`, but never below `min_lr`.

        Parameters
        ----------
        epoch : int
            Index of the current epoch.

        logs : dict, optional
            Dictionary of metrics from the current epoch, including the monitored
            metric and learning rate. If the monitored metric is not in `logs`,
            a warning is issued.

        Warns
        -----
        UserWarning
            If the `monitor` metric is not found in `logs`.

        Notes
        -----
        The new learning rate is assigned directly to the optimizer attribute
        specified during initialization. If `verbose > 0`, a message is printed
        when the learning rate is reduced.
        """
        logs = logs or {}
        if not hasattr(self.model, self.optimizer):
            return
        logs["learning_rate"] = float(
            tf.keras.ops.convert_to_numpy(getattr(self.model, self.optimizer).learning_rate)
        )
        current = logs.get(self.monitor)

        if current is None:
            warnings.warn(
                "Learning rate reduction is conditioned on metric "
                f"`{self.monitor}` which is not available. Available metrics "
                f"are: {','.join(list(logs.keys()))}.",
                stacklevel=2,
            )
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = float(
                        tf.keras.ops.convert_to_numpy(
                            getattr(self.model, self.optimizer).learning_rate
                        )
                    )
                    if old_lr > np.float32(self.min_lr):
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        getattr(self.model, self.optimizer).learning_rate = new_lr
                        if self.verbose > 0:
                            tf.print(
                                f"\nEpoch {epoch + 1}: "
                                "ReduceLROnPlateau reducing "
                                f"{self.optimizer} learning rate to {new_lr}."
                            )
                        self.cooldown_counter = self.cooldown
                        self.wait = 0

    def in_cooldown(self):
        """
        Check if the cooldown period is active.

        Returns
        -------
        bool
            True if the cooldown counter is greater than zero, indicating that
            the cooldown period is still active. False otherwise.

        Examples
        --------
        >>> obj = ReduceLROnPlateau()
        >>> obj.cooldown_counter = 5
        >>> obj.in_cooldown()
        True

        >>> obj.cooldown_counter = 0
        >>> obj.in_cooldown()
        False
        """
        return self.cooldown_counter > 0
