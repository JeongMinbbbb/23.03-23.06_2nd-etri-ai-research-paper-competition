import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization, Dropout, Activation, GlobalAveragePooling1D, concatenate

from keras.utils import np_utils
from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau

from sklearn.preprocessing import LabelEncoder


class MRCnnModel():

  def __init__(self):
        # 라벨 인코딩 객체 생성
        self.label_encoder = LabelEncoder()

        #모델 학습에 필요한 Callback
        self.reLR = ReduceLROnPlateau(patience = 3,verbose = 1,factor = 0.5) 
        self.es = EarlyStopping(monitor='val_loss', patience=10, mode='min')
        self.checkpoint = ModelCheckpoint("./model_checkpoint.h5", 
                                    monitor='val_loss', 
                                    verbose=1, 
                                    save_best_only=True, 
                                    save_weights_only=False, 
                                    mode='auto')

        ######## 모델 구조
        # 입력 데이터의 shape: (60, 7)
        input_layer = Input(shape=(60, 7))

        # model1
        conv1_1 = Conv1D(filters=128, kernel_size=2, strides=1, padding = 'same', activation='relu')(input_layer)
        norm1_1 = BatchNormalization()(conv1_1)
        pool1_1 = MaxPooling1D(2)(norm1_1)
        conv1_2 = Conv1D(filters=128, kernel_size=2, strides=1, padding = 'same', activation='relu')(pool1_1)
        norm1_2 = BatchNormalization()(conv1_2)
        pool1_2 = MaxPooling1D(2)(norm1_2)
        drop1 = Dropout(0.2)(pool1_2)

        # model2
        conv2_1 = Conv1D(filters=128, kernel_size=3, strides=1, padding = 'same', activation='relu')(input_layer)
        norm2_1 = BatchNormalization()(conv2_1)
        pool2_1 = MaxPooling1D(2)(norm2_1)
        conv2_2 = Conv1D(filters=128, kernel_size=3, strides=1, padding = 'same', activation='relu')(pool2_1)
        norm2_2 = BatchNormalization()(conv2_2)
        pool2_2 = MaxPooling1D(2)(norm2_2)
        drop2 = Dropout(0.2)(pool2_2)

        # model3
        conv3_1 = Conv1D(filters=128, kernel_size=5, strides=1, padding = 'same', activation='relu')(input_layer)
        norm3_1 = BatchNormalization()(conv3_1)
        pool3_1 = MaxPooling1D(2)(norm3_1)
        conv3_2 = Conv1D(filters=128, kernel_size=5, strides=1, padding = 'same', activation='relu')(pool3_1)
        norm3_2 = BatchNormalization()(conv3_2)
        pool3_2 = MaxPooling1D(2)(norm3_2)
        drop3 = Dropout(0.2)(pool3_2)

        # merge
        merged = concatenate([drop1, drop2, drop3])

        # CNN
        conv0_3 = Conv1D(filters=128, kernel_size=3, strides=1, activation='relu')(merged)
        norm0_3 = BatchNormalization()(conv0_3)

        conv0_4 = Conv1D(filters=128, kernel_size=3, strides=1, activation='relu')(norm0_3)
        norm0_4 = BatchNormalization()(conv0_4)

        drop4 = Dropout(0.2)(norm0_4)

        pool0_4 = GlobalAveragePooling1D()(drop4)

        # MLP를 통한 분류
        dense1 = Dense(units=128, activation='relu')(pool0_4)
        dense2 = Dense(units=32, activation='relu')(dense1)
        output_layer = Dense(units=5, activation='softmax')(dense2)
        ######## 모델 구조 끝

        # 모델 구성
        adam = keras.optimizers.Adam()
        self.model_cnn = Model(inputs=input_layer, outputs=output_layer)
        self.model_cnn.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['accuracy'])

  def Encode(self, x): # 라벨 -> 숫자
        return self.label_encoder.fit_transform(x)
  
  def Decode(self, x): # 숫자 -> 라벨
        return self.label_encoder.inverse_transform(x)
        
  def fit(self, x_train, encoded_y_train ,x_valid, encoded_y_valid, epochsize = 1000, savename = "model_cnn"):
      self.epochsize = epochsize

      #target 변수 one-hot encoding
      encoded_y_train_hot = np_utils.to_categorical(encoded_y_train) # 3 -> [0,0,0,1,0, ..]
      encoded_y_valid_hot = np_utils.to_categorical(encoded_y_valid)

      history = self.model_cnn.fit(x_train, encoded_y_train_hot, epochs = self.epochsize, batch_size=64,
              validation_data=(x_valid, encoded_y_valid_hot), callbacks=[self.es, self.reLR, self.checkpoint])

      # 모델 저장
      self.model_cnn.save("./" + savename + ".h5")

      # 학습 그래프 출력
      y_vloss = history.history['val_loss']
      y_loss = history.history['loss']

      x_len = np.arange(len(y_loss))
      plt.plot(x_len, y_vloss, marker='.', c='red', label="Validation-set Loss")
      plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")

      plt.legend(loc='upper right')
      plt.grid()
      plt.xlabel('epoch')
      plt.ylabel('loss')
      plt.show()

      return history

  def predict(self, x):
    return self.label_encoder.inverse_transform(np.argmax(self.model_cnn.predict(x), axis=1))

  def make_SP_ts(self, x, visual_target, where = "other_indoor"):
    pred_lst = self.predict(x)
    N = len(pred_lst)
    visual_target_list = visual_target["place"].values
    
    timestamp_ind = []
    for i in range(N):
      if (pred_lst[i] == "home")&(visual_target_list[i] == where):
        timestamp_ind.append(i)

    return visual_target.loc[timestamp_ind,"timestamp_large"]

  def Load_Model(self, path):
      self.model_cnn = load_model(path)
      