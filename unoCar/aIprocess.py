import torch
import cv2
from numpy import random
import numpy as np
from urllib.request import urlopen
from keras.models import load_model
import threading
import time
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

ip = '192.168.137.3'
stream = urlopen('http://' + ip +':81/stream')
buffer = b''
urlopen('http://' + ip + "/action?go=speed40")


#keras 모델불러오기
motor_model = load_model(r"d:\\workspaces\\arduino\\my_model\\keras_model.h5", compile=False)
class_names = open(r"d:\\workspaces\\arduino\\my_model\\labels.txt", "r").readlines()


# YOLOv5 모델 정의
img_model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')

car_state2 = "go"
thread_frame = None  
image_flag = 0
thread_image_flag = 0
img = None

def process():
    
    def yolo_thread():
        global image_flag,thread_image_flag,frame, thread_frame, car_state2
        while True:
            if image_flag == 1:
                thread_frame = frame
                
                # 이미지를 모델에 입력
                results = img_model(thread_frame)

                # 객체 감지 결과 얻기
                detections = results.pandas().xyxy[0]

                if not detections.empty:
                    # 결과를 반복하며 객체 표시
                    for _, detection in detections.iterrows():
                        x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']].astype(int).values
                        label = detection['name']
                        conf = detection['confidence']

                                    
                        
                        if "stop" in label and conf > 0.75:
                            print("stop")
                            car_state2 = "stop"
                        elif "speed40" in label and conf > 0.75:
                            print("speed40")
                            car_state2 = "go"
                            urlopen('http://' + ip + "/action?go=speed40")
                        elif "speed60" in label and conf > 0.75:
                            print("speed60")
                            car_state2 = "go"
                            urlopen('http://' + ip + "/action?go=speed80")
                        elif "farm" in label and conf > 0.75:
                            print("farm")
                            car_state2 ="stop"
                        elif "storage" in label and conf >0.75:
                            print("storage")
                            car_state2 ="stop"
                            urlopen('http://' + ip + "/action?go=speed0")
                            time.sleep(3)
                            car_state2 ="go"
                            urlopen('http://' + ip + "/action?go=speed40")    
                            
                        # 박스와 라벨 표시
                        color = [int(c) for c in random.choice(range(256), size=3)]
                        cv2.rectangle(thread_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(thread_frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                thread_image_flag = 1
                image_flag = 0
                
    # 데몬 스레드를 생성합니다.
    t1 = threading.Thread(target=yolo_thread)
    t1.daemon = True 
    t1.start()


    def image_process_thread():
        global img
        global image_flag
        while True:
            if image_flag == 1:
                img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
                img = (img / 127.5) - 1

                # Predict the img_model
                prediction = motor_model.predict(img)
                index = np.argmax(prediction)
                class_name = class_names[index]
                confidence_score = prediction[0][index]
                percent = int(str(np.round(confidence_score * 100))[:-2])


                if "go" in class_name[2:] and car_state2=="go" and percent >= 95:
                    #print("직진:",str(np.round(confidence_score * 100))[:-2],"%")
                    urlopen('http://' + ip + "/action?go=forward")
                    
                elif "left" in class_name[2:] and car_state2=="go" and percent >= 95:
                    #print("왼쪽:",str(np.round(confidence_score * 100))[:-2],"%")
                    urlopen('http://' + ip + "/action?go=left")
                    
                elif "right" in class_name[2:] and car_state2=="go" and percent >= 95:
                    #print("오른쪽:",str(np.round(confidence_score * 100))[:-2],"%")
                    urlopen('http://' + ip + "/action?go=right")
            
                elif car_state2=="stop":
                    urlopen('http://' + ip + "/action?go=stop")
            
                    
                image_flag = 0

                
    # 데몬 스레드를 생성합니다.
    t2 = threading.Thread(target=image_process_thread)
    t2.daemon = True 
    t2.start()


    urlopen('http://' + ip + "/action?go=stop")
    cv2.destroyAllWindows()
    
