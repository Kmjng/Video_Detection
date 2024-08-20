# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:17:12 2024

@author: minjeong
"""

import numpy as np
import csv

def read_csv(path):
  width = 34
  height = 26
  dims = 1

  with open(path,'r') as f:
    #csv 파일 -> list 
    reader = csv.DictReader(f)
    rows = list(reader)

  #imgs 이미지 배열, tgs 레이블 배열 객체
  imgs = np.empty((len(list(rows)),height,width, dims),dtype=np.uint8)
  tgs = np.empty((len(list(rows)),1))

  for row,i in zip(rows,range(len(rows))):
    #리스트 행을 배열에 저장
    img = row['image']
    img = img.strip('[').strip(']').split(', ')
    im = np.array(img,dtype=np.uint8)
    im = im.reshape((height, width))
    im = np.expand_dims(im, axis=2)
    imgs[i] = im

    #기존 레이블 open, close -> 1, 0 
    tag = row['state']
    if tag == 'open':
      tgs[i] = 1
    else:
      tgs[i] = 0

  # dataset shuffle
  index = np.random.permutation(imgs.shape[0])
  imgs = imgs[index]
  tgs = tgs[index]

  return imgs, tgs # 변환된 데이터셋(이미지, 태그) 반환