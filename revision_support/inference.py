"""Reuse a compiled inference graph without Keras predict's data adapter."""


def predict(model, inputs):
    import tensorflow as tf
    function = getattr(model, '_mira_inference_function', None)
    if function is None:
        signature = tf.TensorSpec(shape=(None,) + tuple(model.input_shape[1:]), dtype=model.inputs[0].dtype)
        function = tf.function(lambda values: model(values, training=False),
                               input_signature=[signature])
        model._mira_inference_function = function
    return function(tf.convert_to_tensor(inputs, dtype=model.inputs[0].dtype)).numpy()
