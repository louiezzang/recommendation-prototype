import math
import tensorflow as tf

queries = tf.ragged.constant([['Who', 'is', 'Dan', 'Smith'],
                              ['Pause'],
                              ['Will', 'it', 'rain', 'later', 'today']])

# Create an embedding table.
num_buckets = 1024
embedding_size = 4
embedding_table = tf.Variable(
    tf.random.truncated_normal([num_buckets, embedding_size],
                               stddev=1.0 / math.sqrt(embedding_size)))

# Look up the embedding for each word.
word_buckets = tf.strings.to_hash_bucket_fast(queries, num_buckets)
word_embeddings = tf.ragged.map_flat_values(
    tf.nn.embedding_lookup, embedding_table, word_buckets)                  # ①

# Add markers to the beginning and end of each sentence.
marker = tf.fill([queries.nrows(), 1], '#')
print("marker =", marker)
padded = tf.concat([marker, queries, marker], axis=1)                       # ②
print("padded =", padded)

# Build word bigrams & look up embeddings.
bigrams = tf.strings.join([padded[:, :-1],
                           padded[:, 1:]],
                          separator='+')                                # ③
print("bigrams =", bigrams)

bigram_buckets = tf.strings.to_hash_bucket_fast(bigrams, num_buckets)
bigram_embeddings = tf.ragged.map_flat_values(
    tf.nn.embedding_lookup, embedding_table, bigram_buckets)                # ④

# Find the average embedding for each sentence
all_embeddings = tf.concat([word_embeddings, bigram_embeddings], axis=1)    # ⑤
avg_embedding = tf.reduce_mean(all_embeddings, axis=1)                      # ⑥
print(avg_embedding)

