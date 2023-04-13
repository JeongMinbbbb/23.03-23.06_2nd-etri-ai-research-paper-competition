#필요 패키지 import
import pandas as pd
import numpy as np
import folium
import os
import tqdm

def get_gps(base_path, fold_name, user_name, duplicated = True):
  df_lst = [] # DataFrame 저장용 List
  if duplicated:
    path_1 = base_path + "\\" + fold_name + "\\" + fold_name + "\\" + user_name # 기본 경로에서 한 단계씩 폴더 접근
  else:
    path_1 = base_path + "\\" + fold_name + "\\" + user_name# 기본 경로에서 한 단계씩 폴더 접근

  ts_list = os.listdir(path_1)
  for t in tqdm.tqdm(ts_list):
      path_2 = path_1 + "\\" + t + "\\" + "mGps" # Gps 센서 데이터만 모으기
      sen_list = os.listdir(path_2)        
      for s in sen_list:
          tmp = pd.read_csv(path_2 + "\\" + s)
          tmp["user"] = user_name
          tmp["timestamp_large"] = s[:-4]
          df_lst.append(tmp)

  df_tmp = pd.concat(df_lst) # 병합
  df_tmp.reset_index(drop=True)
  df_tmp["timestamp_large"] = df_tmp["timestamp_large"].astype(int)

  return df_tmp



def diplay_SP(df, visual_ind, make_html = True):
  gps_df = df.loc[df["timestamp_large"].isin(visual_ind), ["lat","lon"]] #원하는 위치의 위도, 경도만 가져오기
  center = [np.mean(gps_df["lat"]), np.mean(gps_df["lon"])] # 지도 중심 설정

  m = folium.Map(location=center, zoom_start=12)

  for i, j in gps_df.values:
      folium.Circle(location = [i, j],
                          radius = 5, color='#FF2D00', fill_color='#F38168').add_to(m)
  if make_html:
    m.save('map.html')
  
  return m

