#필요 패키지 import
import pandas as pd
import numpy as np
import os
import tqdm


def make_user_df(base_path, duplicated = True):
  sen_lst = ['e4Acc', 'e4Bvp', 'e4Eda', 'e4Hr', 'e4Temp'] #필요 센서 리스트
  df_lst = {}
  for i in sen_lst:
      df_lst[i] = []

  fold_name = ["user01-06","user07-10","user11-12","user21-25","user26-30"] #데이터 경로
  for i in tqdm.tqdm(fold_name):
    if duplicated:
      path_1 = base_path + "\\" + i + "\\" + i #기본 경로에서 한 단계씩 폴더 접근
    else:
      path_1 = base_path + "\\" + i #기본 경로에서 한 단계씩 폴더 접근
    user_list = os.listdir(path_1)

    for u in user_list:
        path_2 = path_1 + "\\" + u
        ts_list = os.listdir(path_2)
        for t in ts_list:
            for sen in sen_lst: #원하는 E4 생체 센서 데이터만
                path_3 = path_2 + "\\" + t + "\\" + sen
                sen_list = os.listdir(path_3)        
                for s in sen_list: 
                    tmp = pd.read_csv(path_3 + "\\" + s)      
                    tmp["timestamp"] = np.trunc(tmp["timestamp"]) #1초 내에 여러번 측정된 센서의 경우 평균으로 요약
                    tmp2 = tmp.groupby("timestamp").mean()
                    tmp2.columns = sen + "_" + tmp2.columns
                    tmp2["user"] = u
                    tmp2["timestamp_large"] = s[:-4] #파일명에서 .csv만 빼고 저장
                    df_lst[sen].append(tmp2)

  for i in sen_lst:
    try:
        tt = pd.merge(pd.concat(df_lst[i]).reset_index(), tt, how = 'inner', on = ["user","timestamp_large","timestamp"]) #user, timestamp 일치하는 경우 merge
    except:
        tt = pd.concat(df_lst[i]).reset_index() #Error가 발생한 경우는 첫 단계일 때, 따라서 tt를 새로 지정

  tt2 = tt.dropna()
  tmp = tt2.value_counts(["user","timestamp_large"]).reset_index().rename(columns={0:"count"})
  tmp2 = pd.merge(tt2,tmp, how='left', on=["user","timestamp_large"])
  tmp2["timestamp_large"] = tmp2["timestamp_large"].astype(int)
  #timestamp_large 기준 60개 (= 1분 = 60초)인 경우만 추출

  return tmp2[tmp2["count"] == 60].drop("count", axis=1)




def make_label_df(base_path, duplicated=True):
  label_list = []
  fold_name = ["user01-06","user07-10","user11-12","user21-25","user26-30"] #데이터 경로 #데이터 경로
  for i in tqdm.tqdm(fold_name):
    if duplicated:
      path_1 = base_path + "\\" + i + "\\" + i #기본 경로에서 한 단계씩 폴더 접근
    else:
      path_1 = base_path + "\\" + i#기본 경로에서 한 단계씩 폴더 접근
    user_list = os.listdir(path_1)
    for u in user_list:
        path_2 = path_1 + "\\" + u
        ts_list = os.listdir(path_2)
        for t in ts_list: #timestamp 폴더 별 label.csv 존재
            tmp = pd.read_csv(path_2 + "\\" + t + "\\" + t + "_label.csv")
            tmp["user"] = u
            tmp["timestamp"] = t
            label_list.append(tmp)
  tmp2 = pd.concat(label_list)
  tmp2["timestamp"] = tmp2["timestamp"].astype(int)
  return tmp2