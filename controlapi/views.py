import cv2
import torch
from numpy import random
import numpy as np
from flask import (
    Blueprint,
    render_template,
    request,
    Response,
    redirect,
    url_for,
    stream_with_context,
)
from threading import Thread
from apps.object_detection import ObjectDetector
import cx_Oracle
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from apps.controlapi.models import Unofarm, Video
import os
import smtplib
from email.mime.text import MIMEText

username = 'uno'
password = 'uno'
dsn = 'localhost/xe'
model = torch.hub.load('apps/controlapi/yolov5', 'custom', path='apps/controlapi/best4.pt', source='local', force_reload=True)
if torch.cuda.is_available():
    model = model.cuda()

from apps.Streamer import Streamer

streamer = Streamer()
src = 0

# 데이터베이스 연결 설정
engine = create_engine(f'oracle://{username}:{password}@{dsn}')
Session = sessionmaker(bind=engine)
db_session = Session()

last_detection_time = None
detection_interval = 0.5
recording_start_time = None
last_detection = None
recording_duration = 9  # 녹화 시간 (초)
save_path = 'C:/control/attach'  # 영상 저장 경로

# Blueprint로 controlapi 앱을 생성한다.
controlapi = Blueprint(
    "controlapi",
    __name__,
    template_folder="templates",
    static_folder="static",
)
# 다음번째 dno를 가져오기 위한 함수 정의
def get_next_dno():
    last_dno = db_session.query(Unofarm.DNO).order_by(Unofarm.DNO.desc()).first()
    if last_dno is None:
        return 1
    else:
        return last_dno[0] + 1
    
def get_next_vno():
    last_vno = db_session.query(Video.VNO).order_by(Video.VNO.desc()).first()
    if last_vno is None:
        return 1
    else:
        return last_vno[0] + 1
    
def send_danger_email(label, time):
    # 이메일 설정
    smtp_server = 'smtp.gmail.com'  # 이메일 서버 주소
    smtp_port = 589  # 이메일 서버 포트
    smtp_username = 'xxx@gmail.com'  # 발신자 이메일 주소
    smtp_password = 'xxxxxxxxxxxxxx'  # 발신자 이메일 비밀번호
    recipient = 'zzz@gmail.com'  # 수신자 이메일 주소

    # 이메일 내용 작성
    subject = '위험 객체 감지 알람'
    body = f'위험 객체 ({label}) 감지 시간: {time}'
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = smtp_username
    msg['To'] = recipient

    try:
        # 이메일 서버에 연결하고 이메일 전송
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.sendmail(smtp_username, recipient, msg.as_string())
        print('위험 객체 감지 알람 이메일을 성공적으로 전송했습니다.')
    except Exception as e:
        print('이메일 전송 중 오류가 발생했습니다:', str(e))

def stream_gen(src):
    global last_detection_time, recording_start_time, last_detection
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    detected_labels = []
    video_writer = None
    try:
        # streamer 객체의 run() 메서드를 호출하여 지정된 소스에서 스트리밍을 시작한다.
        # 이 소스는 함수에 전달된 src 매개변수에 의해 결정된다.
        streamer.run(src)

        # 스트리밍을 계속해서 수행
        while True:
            # streamer 객체의 bytescode() 메서드를 사용하여 스트리밍 소스에서 현재 프레임을 가져온다.
            # 이 프레임은 이미지 데이터의 bytes 형식으로 반환된다.
            frame_bytes = streamer.bytescode()
            # bytes 형식의 이미지 프레임을 가져옴
            frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
            # bytes를 NumPy 배열로 변환
            frame = cv2.imdecode(frame_np, flags=cv2.IMREAD_COLOR)
            # streamer 객체의 bytescode() 메서드를 사용하여 스트리밍 소스에서 현재 프레임을 가져온다.
            # 이 프레임은 이미지 데이터의 bytes 형식으로 반환된다.
            # 이미지를 모델에 입력
            results = model(frame)
            # 객체 감지 결과 얻기
            detections = results.pandas().xyxy[0]
            if not detections.empty:
                current_time = datetime.now()
                if last_detection is None or (current_time - last_detection).total_seconds() >= detection_interval:
                    # 객체 감지 시간 업데이트
                    last_detection = current_time
                # 결과를 반복하며 객체 표시
                    for _, detection in detections.iterrows():
                        x1, y1, x2, y2 = detection[['xmin', 'ymin', 'xmax', 'ymax']].astype(int).values
                        label = detection['name']
                        conf = detection['confidence']
                        print(f'[Unofarm] Detected: {label} ({conf:.2f})')

                        # 감지된 객체를 리스트에 추가
                        if label not in detected_labels:
                            detected_labels.append(label)
                            
                        # 박스와 라벨 표시
                        color = [int(c) for c in random.choice(range(256), size=3)]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    
                    
                    # DB 저장 시간 간격 확인
                if last_detection_time is None or (current_time - last_detection_time).total_seconds() >= 10:
                        # 다음 dno 값 가져오기
                        next_dno = get_next_dno()

                        # Unofarm 객체 생성 및 저장
                        unofarm = Unofarm(
                            DNO=next_dno,
                            D_LABEL=', '.join(detected_labels),
                            D_TIME=current_time,
                            ID='nongjang',
                            D_STATUS='Danger' if any(label in ['person', 'boar'] for label in detected_labels) else 'Safe'
                        )
                        db_session.add(unofarm)
                        db_session.commit()
                        if unofarm.D_STATUS == 'Danger':
                            send_danger_email(unofarm.D_LABEL, unofarm.D_TIME)
                        last_detection_time = current_time
                        detected_labels = []
                        
                        # 영상 녹화 시작
                        if video_writer is None:
                            recording_start_time = current_time
                            file_name = f"Video {current_time.strftime('%Y.%m.%d.%H.%M.%S')}"
                            video_path = os.path.join(save_path, file_name + ".mp4")
                            print(video_path)
                            video_writer = cv2.VideoWriter(video_path, fourcc, 4.0, (1920,1080))
                    
            # 영상 녹화
            if video_writer is not None:
                video_writer.write(frame)
                if (current_time - recording_start_time).total_seconds() >= recording_duration:
                    video_writer.release()
                    video_writer = None
                    
                    # Video 객체 생성 및 저장
                    video = Video(
                        VNO=get_next_vno(),
                        V_TITLE=file_name,
                        PATHUPLOAD=save_path,
                        FILETYPE='.mp4',
                        REGDATE=recording_start_time,
                        DNO=next_dno
                    )
                    db_session.add(video)
                    db_session.commit()
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            # 제너레이터에서 스트리밍 데이터를 반환한다.
            # 각 프레임은 'multipart/x-mixed-replace' 형식의 멀티파트 응답으로 클라이언트에게 전송된다.
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
            )

    # 제너레이터가 종료될 때 발생하는 예외를 처리한다.
    # 이 예외는 주로 제너레이터 함수가 종료되기 전에 처리해야 하는 정리 작업을 수행하기 위해 사용된다.
    # 어떤 이유로든 스트리밍이 종료되면 제너레이터가 종료
    except GeneratorExit:
        print('[Unofarm]', 'disconnected stream')
        if video_writer is not None:
            video_writer.release()
        streamer.stop()

@controlapi.route("/stream")
def stream():
    try:
        return Response(
            stream_with_context(stream_gen(src)),  # ObjectDetector의 run() 메서드를 스트리밍 콘텍스트와 함께 실행
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        print('[Unofarm] ', 'stream error : ', str(e))
 
