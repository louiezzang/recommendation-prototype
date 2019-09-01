import numpy as np
import tensorflow as tf

matrix = np.random.random([1024, 64])  # 64-dimensional embeddings
print("**** matrix=", matrix)
ids = np.array([0, 5, 17, 33])
print("**** embedding matrix=", matrix[ids])  # prints a matrix of shape [4, 64]


params = tf.constant([10,20,30,40])
ids = tf.constant([0,1,2,3])
print(tf.nn.embedding_lookup(params,ids))


params1 = tf.constant([1,2])
params2 = tf.constant([10,20])
ids = tf.constant([2,0,2,1,2,3])
result = tf.nn.embedding_lookup([params1, params2], ids)
print(result)
