import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import string
import gensim
from gensim.models import Word2Vec, KeyedVectors
import keras
from keras.layers.core import Reshape, Flatten
from keras import regularizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Embedding, Flatten, GlobalMaxPool1D, Dropout, Conv1D, LSTM, MaxPooling1D, concatenate
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.models import Model
from keras.initializers import RandomUniform, glorot_uniform
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, hamming_loss, f1_score
import matplotlib.pyplot as plt

import io
import random
import joblib

def tokenize_text(vocab_size, reviews,maxlen):
    tokenizer = Tokenizer(num_words=vocab_size, lower=True)
    tokenizer.fit_on_texts(reviews)
    sequences = tokenizer.texts_to_sequences(reviews)
    x = pad_sequences(sequences, maxlen=maxlen)
    return x, tokenizer
  
def createEmbeddingMatrix(word_index, vocab_size, dim, word_vectors):
    EMBEDDING_DIM=dim
    vocabulary_size=min(len(word_index)+1,vocab_size)
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i>=vocab_size:
            continue
        try:
            embedding_vector = word_vectors[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)
    return embedding_matrix
  
def create_conv_layers(num_filters, filter_sizes, embedding, conv_activation):
  conv_layers = []
  for s in filter_sizes:
    conv = Conv1D(num_filters, s, activation=conv_activation, kernel_initializer=glorot_uniform(seed=random.seed(7)), kernel_regularizer=regularizers.l2(0.01))(embedding)
    conv_layers.append(conv)
  return conv_layers

def max_pools(maxlen, filter_sizes, conv_layers):
  pools = []
  for i in range(len(conv_layers)):
    pool = GlobalMaxPool1D()(conv_layers[i])
    pools.append(pool)
  return pools
  
def create_cnn_model(filter_sizes, num_filters, embedding_matrix, conv_activation, dense_units, embedding_dim, vocabulary_size, maxlen, num_classes):
    drop = 0.5
    
    inputs = Input(shape=(maxlen,))
    embedding_layer = Embedding(vocabulary_size,
                            embedding_dim,
                            weights=[embedding_matrix],
                            trainable=True)
    embedding = embedding_layer(inputs)
    
    convs = create_conv_layers(num_filters, filter_sizes, embedding, conv_activation)
    pools = max_pools(maxlen, filter_sizes, convs)
    if (len(pools)<2):
      dense1 = Dense(dense_units, kernel_initializer=RandomUniform(seed=random.seed(7)))(pools[0])
    else:
      merged_tensor = concatenate(pools, axis=1)
      dense1 = Dense(dense_units, kernel_initializer=RandomUniform(seed=random.seed(7)))(merged_tensor)
  
    dropout = Dropout(drop)(dense1)
    
    output = Dense(units=num_classes, activation='sigmoid',kernel_regularizer=regularizers.l2(0.01))(dropout)
    
    # this creates a model that includes
    model = Model(inputs, output)
    #print(model.summary())
    return model
  
def create_cnn_lstm_model(filter_sizes, num_filters, embedding_matrix, conv_activation, dense_units, embedding_dim, vocabulary_size, maxlen, num_classes):
    filter_sizes = filter_sizes
    num_filters = num_filters
    drop = 0.5
    
    inputs = Input(shape=(maxlen,))
    embedding_layer = Embedding(vocabulary_size,
                            embedding_dim,
                            weights=[embedding_matrix],
                            trainable=True)
    embedding = embedding_layer(inputs)
    
    convs = create_conv_layers(num_filters, filter_sizes, embedding, conv_activation)
    pools = max_pools(maxlen, filter_sizes, convs)
    merged_tensor = concatenate(pools, axis=1)
    
    dense1 = Dense(dense_units, kernel_initializer=RandomUniform(seed=random.seed(7)))(merged_tensor)    
    dropout = Dropout(drop)(dense1)
    lstm_1 = LSTM(64)(dropout)
    output = Dense(units=num_classes, activation='sigmoid',kernel_regularizer=regularizers.l2(0.01))(lstm_1)
    
    # this creates a model that includes
    model = Model(inputs, output)
    #print(model.summary())
    return model

  
def fit_cnn(x_train, y_train, x_val, y_val, filters, window_sizes, embedding_matrix, conv_activation, dense_units, epochs, filepath, earlystop):
  model = None
  model = create_cnn_model(window_sizes, filters, embedding_matrix, conv_activation, dense_units, 400, 5000, 350, 1)
  adam = Adam(lr=1e-3)
  model.compile(loss='binary_crossentropy', metrics=['accuracy'],
              optimizer=adam)
  if earlystop:
    callbacks = [EarlyStopping(patience=4),
            ModelCheckpoint(filepath=filepath, save_best_only=True)]
  else:
    callbacks = [ModelCheckpoint(filepath=filepath)]
    
  t0 = time()  
  results = model.fit(x_train, y_train, batch_size=32, epochs=epochs, callbacks=callbacks, verbose=0, validation_data = (x_val, y_val))
  training_time = time()-t0
  
  cnn_model = keras.models.load_model(filepath)
  metrics = cnn_model.evaluate(x_val, y_val)
  
  y_pred = cnn_model.predict(x_val)
  y_pred_bool = np.array([[0 if x <= 0.5 else 1 for x in arr] for arr in y_pred])
  f1_macro = f1_score(y_val, y_pred_bool, average='macro')
  f1_micro = f1_score(y_val, y_pred_bool, average='micro')  
  hamloss = hamming_loss(y_val,y_pred_bool)
  
  return results, metrics, f1_macro, f1_micro, hamloss, training_time

  
  
  return results

def fit_cnn_lstm(x_train, y_train, x_val, y_val, filters, window_sizes, embedding_matrix, conv_activation, dense_units, epochs, filepath):
  model = None
  model = create_cnn_lstm_model(window_sizes, filters, embedding_matrix, conv_activation, dense_units, 400, 5000, 350, 1)
  adam = Adam(lr=1e-3)
  model.compile(loss='binary_crossentropy', metrics=['categorical_accuracy'],
              optimizer=adam)
  callbacks = [EarlyStopping(patience=4),
            ModelCheckpoint(filepath=filepath, save_best_only=True)]
    
  results = model.fit(x_train, y_train, batch_size=32, epochs=epochs, callbacks=callbacks, verbose=1, validation_data = (x_val, y_val))
  
  return results
  
def get_output_cnn(model, x_train, x_test):
    total_layers = len(model.layers)
    fl_index = total_layers-2
    feature_layer_model = Model(
                     inputs=model.input,
                     outputs=model.get_layer(index=fl_index).output)
    x_train_xg = feature_layer_model.predict(x_train)
    x_test_xg = feature_layer_model.predict(x_test)
    return x_train_xg, x_test_xg