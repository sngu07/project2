import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
import joblib
import av
from collections import deque
import math
import os

# =========================================================
# 1. 설정 및 상수 정의
# =========================================================
# 모델 파일 경로 설정
MODEL_PATH = 'seq_sign_model_final.joblib' 
SEQ_LENGTH = 40

# 지도 및 화면 설정
MAP_WIDTH = 500
CAM_WIDTH = 500
HEIGHT = 700
TOTAL_WIDTH = MAP_WIDTH + CAM_WIDTH

# 색상 (OpenCV는 BGR 순서)
BG_COLOR = (40, 40, 40)
ROOM_COLOR = (200, 200, 200)
ROBOT_COLOR = (50, 50, 255) # 빨강 (BGR)
TEXT_COLOR = (0, 0, 0)

# 맵 위치
ROOMS = {
    'toilet':   (250, 100, 200, 80),
    'room2':    (250, 220, 200, 80),
    'room1':    (250, 340, 200, 80),
    'elevator': (250, 460, 200, 80),
    'home':     (250, 600, 80, 50)
}

# =========================================================
# 2. 모델 로드 (캐싱 사용)
# =========================================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        return None

# =========================================================
# 3. 데이터 추출 함수
# =========================================================
def extract_xyz(hand_lms):
    if hand_lms is None: return [0.0] * 63
    out = []
    for lm in hand_lms.landmark:
        out.extend([lm.x, lm.y, lm.z])
    return out

# =========================================================
# 4. 영상 처리기 클래스
# =========================================================
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model()
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=2, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        
        self.seq_buffer = deque(maxlen=SEQ_LENGTH)
        
        # 로봇 상태 초기화
        self.rx, self.ry = ROOMS['home'][0], ROOMS['home'][1]
        self.tx, self.ty = self.rx, self.ry
        self.speed = 4
        self.status = "Ready"
        self.last_action = "None"
        self.confidence = 0.0

    def update_robot(self):
        dx = self.tx - self.rx
        dy = self.ty - self.ry
        dist = math.hypot(dx, dy)

        if dist > self.speed:
            self.rx += (dx / dist) * self.speed
            self.ry += (dy / dist) * self.speed
        else:
            self.rx = self.tx
            self.ry = self.ty
            if "Moving" in self.status or "Returning" in self.status:
                self.status = "Arrived"

    def draw_map(self, canvas):
        # 배경
        cv2.rectangle(canvas, (0, 0), (MAP_WIDTH, HEIGHT), BG_COLOR, -1)
        # 복도
        cv2.line(canvas, (250, 100), (250, 600), (100, 100, 100), 10)

        # 방 그리기
        for name, (cx, cy, w, h) in ROOMS.items():
            color = ROOM_COLOR
            if name == 'home': color = (255, 100, 100) 
            elif name == 'elevator': color = (100, 255, 255) 
            elif name == 'toilet': color = (255, 255, 100) 

            tl = (cx - w//2, cy - h//2)
            br = (cx + w//2, cy + h//2)
            cv2.rectangle(canvas, tl, br, color, -1)
            cv2.rectangle(canvas, tl, br, (255, 255, 255), 2)
            cv2.putText(canvas, name.upper(), (cx - 40, cy + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)

        # 로봇 그리기
        cv2.circle(canvas, (int(self.rx), int(self.ry)), 15, ROBOT_COLOR, -1)
        cv2.circle(canvas, (int(self.rx), int(self.ry)), 15, (255,255,255), 2)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 캔버스 초기화
        canvas = np.zeros((HEIGHT, TOTAL_WIDTH, 3), dtype=np.uint8)
        
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(image_rgb)

        left_hand, right_hand = None, None
        if result.multi_hand_landmarks:
            for hand_lms, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                self.mp_drawing.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                label = handedness.classification[0].label
                if label == 'Left': left_hand = hand_lms
                else: right_hand = hand_lms
        
        # 데이터 수집 및 예측
        self.seq_buffer.append(extract_xyz(left_hand) + extract_xyz(right_hand))

        if self.model and len(self.seq_buffer) == SEQ_LENGTH:
            input_data = np.array(self.seq_buffer).flatten().reshape(1, -1)
            probs = self.model.predict_proba(input_data)[0]
            idx = np.argmax(probs)
            self.confidence = probs[idx]
            action = self.model.classes_[idx]

            if self.confidence > 0.8:
                self.last_action = action
                if action == 'thankyou':
                    self.tx, self.ty = ROOMS['home'][0], ROOMS['home'][1]
                    self.status = "Returning Home..."
                elif action in ROOMS:
                    self.tx, self.ty = ROOMS[action][0], ROOMS[action][1]
                    self.status = f"Moving to {action.upper()}"

        # 로봇 업데이트 및 그리기
        self.update_robot()
        self.draw_map(canvas)

        # 카메라 화면 배치
        img_resized = cv2.resize(img, (CAM_WIDTH, int(CAM_WIDTH * 0.75)))
        y_offset = (HEIGHT - img_resized.shape[0]) // 2
        canvas[y_offset:y_offset+img_resized.shape[0], MAP_WIDTH:TOTAL_WIDTH] = img_resized

        # 정보 텍스트
        info_x = MAP_WIDTH + 20
        cv2.putText(canvas, f"STATUS: {self.status}", (info_x, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(canvas, f"ACTION: {self.last_action.upper()}", (info_x, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(canvas, f"CONF: {self.confidence*100:.1f}%", (info_x, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(canvas, format="bgr24")

# =========================================================
# 5. Streamlit UI
# =========================================================
st.set_page_config(page_title="AI Robot Navigation", layout="wide")

st.title("🤖 Raspbot AI Sign Language Controller")
st.markdown("""
왼쪽은 **로봇 시뮬레이션 맵**, 오른쪽은 **나의 웹캠**입니다.  
수화를 인식하면 로봇이 해당 장소로 이동합니다.
""")

# 모델 파일 존재 확인
if not os.path.exists(MODEL_PATH):
    st.warning(f"⚠️ `{MODEL_PATH}` 파일이 현재 디렉토리에 없습니다. GitHub 레포지토리에 파일을 업로드했는지 확인해주세요.")
else:
    # WebRTC 스트리머 실행
    ctx = webrtc_streamer(
        key="sign-language",
        video_processor_factory=SignLanguageProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

