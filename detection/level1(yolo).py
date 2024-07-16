# 9-1
import numpy as np
import cv2 as cv
import sys

path = r'C:/ITWILL/OpenCV_1/opencv_1/source/ch9/'

def construct_yolo_v3():
    f=open(path+ 'coco_names.txt', 'r')
    class_names=[line.strip() for line in f.readlines()]
    # YOLOv3 모델 불러오기 
    model=cv.dnn.readNet('C:/ITWILL/OpenCV_1/opencv_1/source/ch9/yolov3.weights','C:/ITWILL/OpenCV_1/opencv_1/source/ch9/yolov3.cfg')
    # 모델에서 레이어 이름들 가져오기 
    layer_names=model.getLayerNames()
    out_layers=[layer_names[i-1] for i in model.getUnconnectedOutLayers()]
    # 출력레이어 인덱스 for문돌려서 가져오고 각각에 레이어 이름 찾아서 out_layers에 저장
    # YOLOv3는 출력레이어가 여러개이고, 객체 탐지 결과 출력 레이어임 
    return model,out_layers,class_names

# construct_yolo_v3 함수 호출하여 반환값 받기
model, out_layers, class_names = construct_yolo_v3()

# class_names 출력하기
print(out_layers) # ['yolo_82', 'yolo_94', 'yolo_106']

'''
yolo_82: 이 레이어는 첫 번째 스케일의 객체 탐지 결과를 출력합니다. 
YOLOv3에서는 입력 이미지를 여러 스케일로 처리하여 다양한 크기의 객체를 탐지할 수 있습니다. yolo_82는 가장 작은 스케일에 해당합니다.

yolo_94: 두 번째 스케일의 객체 탐지 결과를 출력합니다. 
이 레이어는 중간 크기의 객체를 탐지합니다.

yolo_106: 세 번째 스케일의 객체 탐지 결과를 출력합니다. 
가장 큰 크기의 객체를 탐지할 수 있는 레이어입니다.
'''

 
print(model) # < cv2.dnn.Net 0000015D9902B730>

def yolo_detect(img,yolo_model,out_layers):
    height,width=img.shape[0],img.shape[1]
    test_img=cv.dnn.blobFromImage(img,1.0/256,(448,448),(0,0,0),swapRB=True) #  이미지 화소값 0,1로 변환 하고 크기 변환
    
    # test_img를 신경망에 입력 
    yolo_model.setInput(test_img)
    # 신경망 전방 계산(입력 이미지가 네트워크를 통해 순차적으로 전달되면서 다양한 레이어를 통과) 
    output3=yolo_model.forward(out_layers) # 텐서 형태로 저장
    
    box,conf,id=[],[],[]		# 박스, 신뢰도, 부류 번호
    for output in output3: # 세개의 텐서 반복 처리 
        for vec85 in output: # 85차원 벡터( 박스들xywh, 신뢰도o, 부류 확률p1~p80) 
            scores=vec85[5:]
            class_id=np.argmax(scores)
            confidence=scores[class_id]
            if confidence>0.5:	# 신뢰도가 50% 이상인 경우만 취함
                centerx,centery=int(vec85[0]*width),int(vec85[1]*height)
                w,h=int(vec85[2]*width),int(vec85[3]*height)
                x,y=int(centerx-w/2),int(centery-h/2)
                box.append([x,y,x+w,y+h])
                conf.append(float(confidence))
                id.append(class_id)
            
    ind=cv.dnn.NMSBoxes(box,conf,0.5,0.4)
    objects=[box[i]+[conf[i]]+[id[i]] for i in range(len(box)) if i in ind]
    return objects

model,out_layers,class_names=construct_yolo_v3()		# YOLO 모델 생성
colors=np.random.uniform(0,255,size=(len(class_names),3))	# 부류마다 색깔

img=cv.imread('C:/ITWILL/OpenCV_1/opencv_1/source/ch9/busy_street.jpg')
if img is None: sys.exit('파일이 없습니다.')

res=yolo_detect(img,model,out_layers)	# YOLO 모델로 물체 검출

for i in range(len(res)):			# 검출된 물체를 영상에 표시
    x1,y1,x2,y2,confidence,id=res[i]
    text=str(class_names[id])+'%.3f'%confidence
    cv.rectangle(img,(x1,y1),(x2,y2),colors[id],2)
    cv.putText(img,text,(x1,y1+30),cv.FONT_HERSHEY_PLAIN,1.5,colors[id],2)

cv.imshow("Object detection by YOLO v.3",img)

cv.waitKey()
cv.destroyAllWindows()