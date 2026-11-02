"""Reuse GPflow graphs while executing the authors' original fitting functions.

Every date resets every model parameter to the fresh author's initialization.
No warm start, altered optimizer, reduced iterations or approximate likelihood.
"""
import inspect


class GraphPool:
    def __init__(self):
        self.models = {}
        self.optimizers = {}

    def gpr(self, data, kernel, noise_variance=1.0):
        import gpflow
        import tensorflow as tf
        from tensorflow_probability import bijectors as tfb
        import numpy as np
        x, y = data
        cp = hasattr(kernel, 'locations')
        key = (cp, len(x))
        if key not in self.models:
            model = gpflow.models.GPR(data=data, kernel=kernel, noise_variance=noise_variance)
            model.data = (tf.Variable(x, dtype=tf.float64, trainable=False),
                          tf.Variable(y, dtype=tf.float64, trainable=False))
            low = width = None
            if cp:
                low = tf.Variable(float(x[0, 0]), dtype=tf.float64, trainable=False)
                width = tf.Variable(float(x[-1, 0]-x[0, 0]), dtype=tf.float64, trainable=False)
                transform = tfb.Chain([tfb.Chain([tfb.Shift(low), tfb.Scale(width)]), tfb.Sigmoid()])
                model.kernel.locations = gpflow.Parameter(kernel.locations.numpy(), transform=transform, dtype=tf.float64)
            self.models[key] = (model, low, width)
            self.optimizers[id(model)] = gpflow.optimizers.Scipy()
        model, low, width = self.models[key]
        model.data[0].assign(np.asarray(x, dtype=float))
        model.data[1].assign(np.asarray(y, dtype=float))
        if cp:
            low.assign(float(x[0, 0]))
            width.assign(float(x[-1, 0]-x[0, 0]))
        # Capture only the new author's kernel parameters; exclude cached data/bounds.
        if model.kernel is not kernel:
            gpflow.utilities.multiple_assign(model.kernel,
                {name: value.numpy() for name, value in gpflow.utilities.parameter_dict(kernel).items()})
        model.likelihood.variance.assign(noise_variance)
        return model

    def optimizer(self, model):
        return self.optimizers[id(model)]


def install():
    import mom_trans.changepoint_detection as module
    if getattr(module, '_graph_pool', None) is not None:
        return module._graph_pool
    pool = GraphPool()
    module._graph_pool = pool
    module._original_fits = {name: getattr(module, name) for name in ('fit_matern_kernel', 'fit_changepoint_kernel')}
    module.reusable_gpr = pool.gpr
    module.reusable_optimizer = pool.optimizer
    for name, original in module._original_fits.items():
        source = inspect.getsource(original)
        assert source.count('m = gpflow.models.GPR(') == 1
        assert source.count('opt = gpflow.optimizers.Scipy()') == 1
        source = source.replace('m = gpflow.models.GPR(', 'm = reusable_gpr(')
        source = source.replace('opt = gpflow.optimizers.Scipy()', 'opt = reusable_optimizer(m)')
        exec(compile(source, original.__code__.co_filename, 'exec'), module.__dict__)
    return pool


def uninstall():
    import mom_trans.changepoint_detection as module
    for name, original in getattr(module, '_original_fits', {}).items():
        setattr(module, name, original)
    module._graph_pool = None
