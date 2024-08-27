# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:59:01 2024

@author: minjeong
"""

import cv2
import dlib 
import numpy as np
from model import Net2
import torch
from imutils import face_utils

import os
import time
import winsound  
import threading  # 추가된 모듈 (winsound를 프로그램과 비동기식으로 실행하기 위해 추가)


IMG_SIZE = (34,26)

PATH = 'C:/ITWILL/Video_Detection/detection/Pytorch/pytorch_code/weights/quantized_model.pth'
ALERT_FILE =  'C:/ITWILL/Video_Detection/detection/Pytorch/message'
ALERT_SOUND = 'C:/ITWILL/Video_Detection/detection/Pytorch/sound/notify.wav'

 
detector = dlib.get_frontal_face_detector() # 얼굴을 검출하는 dlib의 얼굴 검출기.
predictor = dlib.shape_predictor('C:/ITWILL/Video_Detection/detection/shape_predictor_68_face_landmarks.dat')
# 얼굴의 랜드마크를 예측하는 dlib의 랜드마크 예측기
model = Net2()
#model.load_state_dict(torch.load(PATH)) # pth에 저장된 가중치 업로드 
model = torch.jit.load(PATH) 
model.eval()


def crop_eye(img, eye_points):
  x1, y1 = np.amin(eye_points, axis=0)
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(int)

  eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect


def predict(pred):
  pred = pred.transpose(1, 3).transpose(2, 3)
  outputs = model(pred)
  pred_tag = torch.round(torch.sigmoid(outputs))

  return pred_tag


#---(추가)-----
# 프레임에서 눈이 감지 되었는지 확인 / 로그파일에 메세지 기록
def save_alert_to_file(message, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(os.path.join(folder_path, 'alert.txt'), 'a') as file: 
        file.write(message + '\n')

# 3초 이상 눈 감는 경우 경고음 실행
def play_alert_sound():
    winsound.PlaySound(ALERT_SOUND, winsound.SND_FILENAME)
#---(추가)-----


n_count = 0
alert_playing = False  # 알람이 울리는 상태를 추적하기 위한 변수


# 비디오 캡처 객체를 저장된 비디오 파일로 초기화
video_path = 'C:/ITWILL/Video_Detection/detection/Pytorch/videos/3.mp4'
cap = cv2.VideoCapture(video_path)

# GIF 저장을 위한 프레임 리스트
frames = []

while cap.isOpened():
    ret, img_ori = cap.read()
    if not ret:
        break

    img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)

    img = img_ori.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    faces = detector(gray) # dlib의 얼굴검출기로 grayscale 이미지에서 얼굴 검출

    # (추가) 
    eyes_detected = False

    for face in faces:  # 얼굴 랜드마크 검출 및 눈 영역 추출
        shapes = predictor(gray, face)
        shapes = face_utils.shape_to_np(shapes)
        
        # 눈의 랜드마크 좌표를 기반으로 눈 영역 crop
        eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
        eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

        eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
        eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
        eye_img_r = cv2.flip(eye_img_r, flipCode=1)

        # 모델 예측
        # 눈 이미지 데이터 Pytorch 텐서로 변환 후 모델 입력
        eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)
        eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)

        eye_input_l = torch.from_numpy(eye_input_l)
        eye_input_r = torch.from_numpy(eye_input_r)
        
        # 예측 결과 반환
        pred_l = predict(eye_input_l) # 왼눈 
        pred_r = predict(eye_input_r) # 오른눈 

        # 예측 결과 처리 및 시각화 
        # 예측 결과를 바탕으로 눈 감김 상태를 체크하고 특정 조건을 만족하면 경고 메시지를 화면에 출력
        if pred_l.item() == 0.0 and pred_r.item() == 0.0:
            n_count += 1
            # (추가)
            eyes_detected = True
        else:
            n_count = 0 # 프레임 카운트 초기화 
            # (추가)
            alert_playing = False  # 눈을 떴을 때 알람 상태 초기화
            
        if n_count > 90: # 90 프레임 (~3초)
            if not alert_playing:  # 알람이 울리지 않는 상태에서만 실행
                cv2.putText(img, "Wake up", (120, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # ----(추가)----
                #play_alert_sound()
                threading.Thread(target=play_alert_sound).start()  # 알람을 비동기적으로 실행
                alert_playing = True  # 알람 상태를 활성화
                # ----(추가)----
        
        # visualize
        state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'
        state_r = 'O %.1f' if pred_r > 0.1 else '- %.1f'

        state_l = state_l % pred_l
        state_r = state_r % pred_r
        
        #cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255,255,255), thickness=2)
        #cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255,255,255), thickness=2)
        if n_count > 90:
            cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(0,0,255), thickness=2)
            cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(0,0,255), thickness=2)
            cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        else : 
            cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255,255,255), thickness=2)
            cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255,255,255), thickness=2)
            cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            
        #cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        #cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # ----(추가)-----
    if not eyes_detected:
        save_alert_to_file("눈이 감지되지 않음", ALERT_FILE)
    #----(추가)------
        
    # 프레임을 리스트에 추가
    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))    
        
    cv2.imshow('result', img) # 결과 프레임을 화면에 표시
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# GIF로 저장
import imageio
gif_path = 'C:/Users/minjeong/Documents/itwill/Video_Detection/detection/Pytorch/videos/output.gif'
with imageio.get_writer(gif_path, mode='I', duration=0.1) as writer:
    for frame in frames:
        writer.append_data(frame)



