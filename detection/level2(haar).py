from __future__ import print_function
import cv2 as cv
import argparse

'''
얼굴 탐지나 눈 탐지 등의 객체 검출 작업에서는 대개 회색조 이미지가 사용됩니다. 
이유는 회색조 이미지는 컬러 정보가 제거되어 처리가 더 간단하고 속도가 빠르며, 
빛과 그림자를 감지하는 데 유리하기 때문입니다.

히스토그램 평활화는 이미지의 전역 대비를 향상시켜 세부적인 정보를 더 잘 볼 수 있게 합니다.
특히 얼굴 검출에서는 얼굴이 어두운 부분과 밝은 부분이 섞여 있을 수 있습니다. 
이런 경우 히스토그램 평활화를 적용하면 얼굴의 주요 특징이 더 잘 부각되어 얼굴 검출 정확도를 향상
'''
def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # 입력프레임을 흑백으로 
    frame_gray = cv.equalizeHist(frame_gray) # 히스토그램 평활화 
    #-- 얼굴 탐지
    faces = face_cascade.detectMultiScale(frame_gray)  
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4) # 얼굴에 타원형 그림 그리기 
        faceROI = frame_gray[y:y+h,x:x+w] 
        #-- 얼굴에서 눈 탐지
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
    cv.imshow('Capture - Face detection', frame)
    
# argparse를 사용하여 얼굴과 눈 검출에 사용할 분류기 파일의 경로와 카메라 장치 번호를 받습니다.
parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
#-- 데이터 파일 경로(본인 환경에 맞게 변경)
parser.add_argument('--face_cascade', help='Path to face cascade.', default="C:/Users/user/anaconda3/envs/tf_env/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml")
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default= "C:/Users/user/anaconda3/envs/tf_env/Lib/site-packages/cv2/data/haarcascade_eye_tree_eyeglasses.xml")
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)

args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()
# cv.CascadeClassifier()를 사용하여 얼굴과 눈을 위한 분류기 객체를 초기화합니다.

#-- 비디오 파일 경로 설정
video_path = r"C:/ITWILL/OpenCV_1/opencv_1/face_detection_source/Video_Detection/IMG_2093.mp4"

#-- 비디오 캡처 객체 생성
cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    print('--(!)Error opening video capture')
    exit(0)

# 캡처 객체가 정상적으로 열렸는지 확인
if not cap.isOpened():
    print('--(!)Error opening video capture')
    exit(0)

#-- 비디오 프레임 읽기 및 처리
while True:
    ret, frame = cap.read() # cap.read()를 사용하여 비디오의 각 프레임을 읽음
    if not ret:
        print('--(!) No captured frame -- Break!')
        break
    
    # 얼굴과 눈 감지 함수 호출
    detectAndDisplay(frame) #  얼굴과 눈을 탐지하고 표시

    # 'q' 키를 누르면 종료
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 캡처 객체 해제 및 모든 창 닫기
cap.release()
cv.destroyAllWindows()



'''
#-- 데이터 파일 읽어오기
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)
camera_device = args.camera

#-- 비디오 or 캠 불러오기
cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)

#-- q 입력시 종료
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
'''