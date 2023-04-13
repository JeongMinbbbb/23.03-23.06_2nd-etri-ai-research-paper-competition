#필요 패키지 import
import pandas as pd
import numpy as np
import os
import random

from sklearn.model_selection import train_test_split


def split(sec_user, last_label_df, visual_name = "user25", seed1 = 42, seed2 = 42):
  sec_user = sec_user.dropna()

  #60 count 아닌 행 제거
  tmp = sec_user.value_counts(["user","timestamp_large"]).reset_index().rename(columns = {0 : "count"})
  tmp2 = pd.merge(sec_user, tmp, how = 'left', on = ["user","timestamp_large"])
  sec_user = tmp2[tmp2["count"] == 60].drop("count" , axis = 1)

  last_label_df2 = last_label_df[["user", 'ts', 'place']].rename(columns = {"ts" : "timestamp_large"})
  user_time_list = sec_user.value_counts(["user","timestamp_large"]).reset_index().drop(0, axis=1)
  target_df = pd.merge(user_time_list, last_label_df2, how='left', on = ['user', 'timestamp_large'])
  target_df = target_df.drop_duplicates(subset = ["user","timestamp_large"], keep = False)
  target_df = target_df[target_df["place"].notna()].reset_index(drop=True)

  sec_user_tmp = pd.merge(sec_user, target_df[["user","timestamp_large","place"]], how = 'left', on = ["user","timestamp_large"])
  sec_user = sec_user_tmp[sec_user_tmp["place"].notna()].drop("place", axis = 1).reset_index(drop = True)


  sec_user = sec_user.sort_values(["user", 'timestamp_large', 'timestamp'])
  target_df = target_df.sort_values(["user", 'timestamp_large'])

  # 사후 분석 데이터셋 분리
  visual_df = sec_user[sec_user["user"] == visual_name].copy()
  visual_target = target_df[target_df["user"] == visual_name].copy()

  # 사후 분석 제외 데이터셋
  sec_user_2 = sec_user[sec_user["user"] != visual_name].copy()
  target_df_2 = target_df[target_df["user"] != visual_name].copy()
  

 # 불필요 변수 제거
  sec_user_tmp = sec_user_2.drop(["timestamp", "user", "timestamp_large"], axis=1)
  visual_df_tmp = visual_df.drop(["timestamp", "user", "timestamp_large"], axis=1)


  # 연구 모델에 맞는 input 형식으로 변경
  sec_user_3d = sec_user_tmp.values.reshape(int(len(sec_user_tmp)/60), 60, 7) #train, valid, test X
  x_visual = visual_df_tmp.values.reshape(int(len(visual_df_tmp)/60), 60, 7) #Visual X

  #train, valid, test 용 target 값은 array 형태로 반환
  target_list= target_df_2["place"].values

  # random하게 modeling용, test용 데이터셋 split
  idx = [i for i in range(len(sec_user_3d))]
  random.seed(seed1)
  random.shuffle(idx)
  model_idx = idx[:int(len(idx)*0.8)]
  test_idx = idx[int(len(idx)*0.8):] 

  x_model = sec_user_3d[model_idx]
  y_model = target_list[model_idx]
  x_test = sec_user_3d[test_idx]
  y_test = target_list[test_idx]

  # modeling 데이터셋을 train과 valid 데이터셋으로 분리
  x_train, x_valid, y_train, y_valid = train_test_split(x_model, y_model, test_size=1/8, random_state=seed2, stratify=y_model)
  
  # 결과적으로 train:valid:test = 7:1:2
  return x_train, x_valid, x_test, x_visual, y_train, y_valid, y_test, visual_target