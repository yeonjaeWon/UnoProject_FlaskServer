from flask import Flask, render_template, jsonify, Blueprint, Response, request
import torch
import cv2
import cx_Oracle
from numpy import random
import numpy as np
from urllib.request import urlopen
from keras.models import load_model
import torch
import threading
import time
import os
import datetime
from time import sleep

app = Flask(__name__)

unocar = Blueprint(
    "unocar",
    __name__,
    template_folder="templates", 
    static_folder="static"
)

# ip주소 및 포트 설정
ip = '192.168.137.36'
stream = urlopen('http://' + ip +':81/stream')
buffer = b''

# keras 모델 로드
motor_model = load_model(r"d:\\workspaces\\arduino\\my_model\\keras_model.h5", compile=False)
class_names = open(r"d:\\workspaces\\arduino\\my_model\\labels.txt", "r").readlines()

# YOLOv5 모델 로드
img_model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')

# 차량 상태 및 이미지 플래그 초기화
car_state2 = "go"
thread_frame = None  
image_flag = 0
thread_image_flag = 0
img = None
key = None

# 데이터베이스 연결 정보
username = 'mire001'
password = 'admin'
dsn = '192.168.0.55/xe'

# 데이터베이스 연결 및 데이터 삽입/업데이트 함수
def insert_or_update_database(speed, state):
    global current_suno
    try:
        db_connection = cx_Oracle.connect(username, password, dsn)
        db_cursor = db_connection.cursor()

        select_query = """
            SELECT COUNT(*) FROM AI_TRANS WHERE SUNO = :suno
        """
        db_cursor.execute(select_query, suno=current_suno)
        count = db_cursor.fetchone()[0]

        if count > 0:
            update_query = """
                UPDATE AI_TRANS
                SET TP_SPEED = :speed, TP_STATE = :state, TP_TIME = SYSDATE
                WHERE SUNO = :suno
            """
            db_cursor.execute(update_query, speed=speed, state=state, suno=current_suno)
            print("Data updated in the database successfully.")
        else:
            insert_query = """
                INSERT INTO AI_TRANS (TP_SPEED, TP_STATE, TP_TIME, SUNO)
                VALUES (:speed, :state, SYSDATE, :suno)
            """
            db_cursor.execute(insert_query, speed=speed, state=state, suno=current_suno)
            print("Data inserted into the database successfully.")

        db_connection.commit()

        db_cursor.close()
        db_connection.close()
    except cx_Oracle.Error as error:
        print("Error occurred while inserting or updating data in the database:", error)

def yolo_thread():
    global image_flag, thread_image_flag, frame, thread_frame, car_state2
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
                        time.sleep(1)
                        car_state2 = "go"
                        urlopen('http://' + ip + "/action?go=speed40")
                    elif "speed40" in label and conf > 0.75:
                        print("speed40")
                        car_state2 = "go"
                        insert_or_update_database(40, "운행중")
                        urlopen('http://' + ip + "/action?go=speed40")
                    elif "speed60" in label and conf > 0.75:
                        print("speed60")
                        car_state2 = "go"
                        insert_or_update_database(60, "운행중")
                        urlopen('http://' + ip + "/action?go=speed60")
                    elif "farm" in label and conf > 0.75:
                        print("farm")
                        car_state2 = "stop"
                        insert_or_update_database(0, "복귀완료")
                    elif "storage" in label and conf > 0.75:
                        print("storage")
                        car_state2 = "stop"
                        insert_or_update_database(0, "물품하차중")
                        urlopen('http://' + ip + "/action?go=speed0")
                        time.sleep(3)
                        insert_or_update_database(0, "하차완료")
                        time.sleep(3)
                        car_state2 = "go"
                        insert_or_update_database(40, "복귀중")
                        urlopen('http://' + ip + "/action?go=speed40")
                    
                    # 박스와 라벨 표시
                    color = [int(c) for c in random.choice(range(256), size=3)]
                    cv2.rectangle(thread_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(thread_frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            thread_image_flag = 1
            image_flag = 0

def image_process_thread():
    global img, ip, image_flag, car_state2, motor_model, class_names
    
    while True:
        if image_flag == 1:
            global img
            img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
            img = (img / 127.5) - 1

            # Predict the img_model
            prediction = motor_model.predict(img)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]
            percent = int(str(np.round(confidence_score * 100))[:-2])
            
            if "go" in class_name[2:] and car_state2=="go" and percent >= 95:
                print("직진:",str(np.round(confidence_score * 100))[:-2],"%")
                urlopen('http://' + ip + "/action?go=forward")
                
            elif "left" in class_name[2:] and car_state2=="go" and percent >= 95:
                print("왼쪽:",str(np.round(confidence_score * 100))[:-2],"%")
                urlopen('http://' + ip + "/action?go=left")
                
            elif "right" in class_name[2:] and car_state2=="go" and percent >= 95:
                print("오른쪽:",str(np.round(confidence_score * 100))[:-2],"%")
                urlopen('http://' + ip + "/action?go=right")
        
            elif car_state2=="stop":
                urlopen('http://' + ip + "/action?go=stop")
                
            image_flag = 0
    
@unocar.route('/')
def car():
    return render_template('car.html')

def send_video():
    global thread_frame
    
    while True:
        ret, buffer11 = cv2.imencode('.jpg', thread_frame)
        frame11 = buffer11.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame11 + b'\r\n')

@unocar.route('/video')
def video():
    return Response(send_video(),  mimetype='multipart/x-mixed-replace; boundary=frame')

def video_stream():
    global stream, buffer, thread_image_flag, car_state2, image_flag, frame, img, current_suno

    # 데몬 스레드를 생성합니다.
    t1 = threading.Thread(target=yolo_thread)
    t1.daemon = True 
    t1.start()

    t2 = threading.Thread(target=image_process_thread)
    t2.daemon = True 
    t2.start()

    while True:
        buffer += stream.read(4096)
        head = buffer.find(b'\xff\xd8')
        end = buffer.find(b'\xff\xd9')
        
        if head > -1 and end > -1:
            jpg = buffer[head:end+2]
            buffer = buffer[end+2:]
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            # 프레임 크기 조정
            frame = cv2.resize(img, (640, 480))

            # 아래부분의 반만 자르기
            height, width, _ = img.shape
            img = img[height // 2:, :]

            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

            cv2.imshow("AI CAR Streaming", img)
            
            image_flag = 1
        
            #쓰레드에서 이미지 처리가 완료되었으면
            if thread_image_flag == 1:
                #cv2.imshow('thread_frame', thread_frame)
                thread_image_flag = 0

            db_connection = cx_Oracle.connect(username, password, dsn)
            db_cursor = db_connection.cursor()
            select_query = "SELECT MAX(SUNO) FROM TRANS_R"
            db_cursor.execute(select_query)
            suno = db_cursor.fetchone()[0]
            if suno is None:
                suno = 0  # 또는 적절한 초기값 설정
            db_cursor.close()
            db_connection.close()

            current_suno = suno

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == 32:
                car_state2 = "go"

# 데몬 스레드를 생성합니다.
t3 = threading.Thread(target=video_stream)
t3.daemon = True 
t3.start()