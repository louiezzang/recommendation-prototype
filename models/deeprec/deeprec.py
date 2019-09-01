from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import re
from collections import defaultdict

import tensorflow as tf
from tensorflow import feature_column

from sklearn.model_selection import train_test_split

MAX_USER_CONTEXT_WINDOW_SIZE = 200
BATCH_SIZE = 500
EMBEDDING_SIZE = 128
EPOCHS = 50

dataset_dir = '/Users/youngguebae/Documents/projects/recommendation-engine/datasets/movielens-small'

df_ratings = pd.read_csv(f'{dataset_dir}/ratings.csv')
df_tags = pd.read_csv(f'{dataset_dir}/tags.csv')
df_movies = pd.read_csv(f'{dataset_dir}/movies.csv')

train, test = train_test_split(df_ratings, test_size=0.2)

item_dics = defaultdict(lambda: len(item_dics))
item_ids = df_ratings['movieId'].unique()
for id in item_ids:
    item_dics[int(id)]
n_items = len(item_dics)


def _build_user_feature(df_ratings):
    df_ratings_ = df_ratings.copy()
    df_ratings_ = df_ratings_.sort_values(['userId', 'timestamp'], ascending=[True, False])

    df_user_features = df_ratings_.groupby(
        'userId'
    ).agg(
        {
            'movieId': lambda x: x.unique().tolist(),
            # 'timestamp': 'max'
        }
    )

    df_user_features.rename(columns={'movieId': 'watchedMovies'}, inplace=True)
    df_user_features['watchedMovies'] = df_user_features['watchedMovies'].apply(
        lambda x: x[:MAX_USER_CONTEXT_WINDOW_SIZE])
    return df_user_features


def _parse_item_age(title):
    year = re.search(r'\(([0-9]{4})\)', title)
    if year is not None:
        age = 2019 - int(year.group(1))
    else:
        age = 10
    return age


def _build_dataset(df_ratings, df_user_features, df_movies, shuffle=True, batch_size=32):
    """
    Creates a tf.data dataset from a Pandas Dataframe
    :param df_ratings:
    :param df_user_features:
    :param df_movies:
    :param shuffle:
    :param batch_size:
    :return:
    """
    df_ratings_ = df_ratings.copy()
    df_ratings_['labelId'] = df_ratings_['movieId'].apply(lambda x: item_dics[x])
    df_ratings_ = pd.merge(df_ratings_, df_user_features, on='userId', how='left')
    # Add example age
    df_movies['age'] = df_movies['title'].apply(lambda x: _parse_item_age(x))
    df_ratings_ = pd.merge(df_ratings_, df_movies, on='movieId', how='left')
    df_ratings_.rename(columns={'age': 'exampleAge'}, inplace=True)
    df_input = df_ratings_[['watchedMovies', 'exampleAge', 'labelId']]

    df_input.reset_index()
    labels = df_input.pop('labelId')

    # Add user embedding features
    embedding = tf.Variable(tf.random.uniform([len(item_dics), EMBEDDING_SIZE], -1.0, 1.0))

    x_batch = np.zeros((len(df_input), MAX_USER_CONTEXT_WINDOW_SIZE))
    embedding_mask = np.zeros((len(df_input), MAX_USER_CONTEXT_WINDOW_SIZE))
    word_num = np.zeros(len(df_input))
    line_no = 0
    for row in df_input['watchedMovies'].tolist():
        col_no = 0
        for movie in row:
            if movie in item_dics:
                x_batch[line_no][col_no] = item_dics[movie]
                # print("col_no=", col_no)
                embedding_mask[line_no][col_no] = 1
                col_no += 1
            if col_no >= MAX_USER_CONTEXT_WINDOW_SIZE:
                break
        word_num[line_no] = col_no
        line_no += 1

    tf_x_batch = tf.constant(x_batch, dtype=tf.int32)

    embedding_mask = embedding_mask.reshape(len(df_input), MAX_USER_CONTEXT_WINDOW_SIZE, 1)
    word_num = word_num.reshape(len(df_input), 1)

    input_embedding = tf.nn.embedding_lookup(params=embedding, ids=tf_x_batch)
    project_embedding = tf.divide(tf.reduce_sum(tf.multiply(input_embedding, embedding_mask), 1), word_num)

    example_age = tf.constant(df_input['exampleAge'].values.reshape(len(df_input), 1), dtype=tf.float32)
    if not shuffle:  # Not training
        example_age = tf.zeros((len(df_input), 1), dtype=tf.float32)

    features = tf.concat([project_embedding, example_age], 1)
    print("*** features shape =", features.shape)
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df_input))
    ds = ds.batch(batch_size)
    return ds


df_user_features = _build_user_feature(df_ratings)
train_ds = _build_dataset(train, df_user_features, df_movies, shuffle=True, batch_size=BATCH_SIZE)
test_ds = _build_dataset(test, df_user_features, df_movies, shuffle=False, batch_size=BATCH_SIZE)

# for feature_batch, label_batch in train_ds.take(1):
#     # print('Every feature:', list(feature_batch.keys()))
#     print('A batch of targets:', label_batch)


class RecModel(tf.keras.Model):
    def __init__(self):
        super(RecModel, self).__init__()
        self.d1 = tf.keras.layers.Dense(1024, activation='relu')
        self.d2 = tf.keras.layers.Dense(512, activation='relu')
        self.d3 = tf.keras.layers.Dense(256, activation='relu')
        self.d4 = tf.keras.layers.Dense(n_items, activation='softmax')

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return x


model = RecModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

# optimizer = tf.keras.optimizers.Adam()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
# optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-3)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(features, labels):
    predictions = model(features)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


for epoch in range(EPOCHS):
    for features, labels in train_ds:
        train_step(features, labels)

    for test_features, test_labels in test_ds:
        test_step(test_features, test_labels)

    template = '에포크: {}, 손실: {}, 정확도: {}%, 테스트 손실: {}, 테스트 정확도: {}%'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))


# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(1024, activation='relu'),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.Dense(n_items, activation='softmax')
# ])
#
# model.compile(optimizer='adam', # optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
#               loss='sparse_categorical_crossentropy',
#               metrics=['sparse_categorical_accuracy'])
#
# model.fit(train_ds,
#           epochs=10)
#
# loss, accuracy = model.evaluate(test_ds)
# print("정확도", accuracy)
