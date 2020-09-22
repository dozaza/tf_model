import tensorflow as tf
from tensorflow.keras import layers


def attention_fun(Q, K, scaled_=True, masked_=False):
    attention = tf.matmul(Q, K, transpose_b=True)       # [batch, seq_len, seq_len]
    if scaled_:
        d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
        attention = tf.divide(attention, tf.sqrt(d_k))  # [batch, seq_len, seq_len]
    if masked_:
        raise NotImplementedError
    attention = tf.nn.softmax(attention, axis=-1)       # [batch, seq_len, seq_len]
    return attention


def model_fun(inputs, **params):
    Q = layers.Dense(params['hidden'])(inputs)      # [batch, seq_len, hidden]
    K = layers.Dense(params['hidden'])(inputs)      # [batch, seq_len, hidden]
    V = layers.Dense(params['n_classes'])(inputs)   # [batch, seq_len, n_classes]
    attention = attention_fun(Q, K)                 # [batch, seq_len, seq_len]
    outputs = tf.matmul(attention, V)               # [batch, seq_len, n_classes]
    return outputs


def input_fun(**params):
    data = tf.random.normal([params['batch'], params['seq_length'], params['hidden']])
    return data


if __name__ == '__main__':
    inputs = input_fun(batch=32, seq_length=10, hidden=128)
    outputs = model_fun(inputs, hidden=128, n_classes=2)
    print(inputs.shape)
    print(outputs.shape)
