import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
import joblib
import av
from collections import deque
import math

# =========================================================
# 1. 설정 및 상수 정의
# =========================================================
MODEL_PATH = 'model.joblib'  # 모델 파일 이름 확인!
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

# 맵 위치 (중심 x, 중심 y, 너비, 높이) - Map 영역 기준
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
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
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
# 4. 영상 처리기 클래스 (핵심 로직)
# =========================================================
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        # 모델 및 도구 초기화
        self.model = load_model()
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=2, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        
        # 데이터 버퍼
        self.seq_buffer = deque(maxlen=SEQ_LENGTH)
        
        # 로봇 상태
        self.rx, self.ry = ROOMS['home'][0], ROOMS['home'][1]
        self.tx, self.ty = self.rx, self.ry
        self.speed = 4
        self.status = "Ready"
        self.last_action = "None"
        self.confidence = 0.0

    def update_robot(self):
        # 로봇 이동 로직
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
        # 1. 배경 (왼쪽 절반)
        cv2.rectangle(canvas, (0, 0), (MAP_WIDTH, HEIGHT), BG_COLOR, -1)
        
        # 2. 복도 라인
        cv2.line(canvas, (250, 100), (250, 600), (100, 100, 100), 10)

        # 3. 방 그리기
        for name, (cx, cy, w, h) in ROOMS.items():
            color = ROOM_COLOR
            if name == 'home': color = (255, 100, 100) # Blue-ish (BGR)
            elif name == 'elevator': color = (100, 255, 255) # Yellow
            elif name == 'toilet': color = (255, 255, 100) # Cyan

            # 사각형 그리기
            tl = (cx - w//2, cy - h//2)
            br = (cx + w//2, cy + h//2)
            cv2.rectangle(canvas, tl, br, color, -1)
            cv2.rectangle(canvas, tl, br, (255, 255, 255), 2)
            
            # 텍스트 (OpenCV 폰트)
            cv2.putText(canvas, name.upper(), (cx - 40, cy + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)

        # 4. 로봇 그리기
        cv2.circle(canvas, (int(self.rx), int(self.ry)), 15, ROBOT_COLOR, -1)
        cv2.circle(canvas, (int(self.rx), int(self.ry)), 15, (255,255,255), 2)

    def recv(self, frame):
        # 1. 웹캠 프레임 가져오기 (av -> numpy)
        img = frame.to_ndarray(format="bgr24")
        
        # 거울 모드 해제 (학습 데이터와 맞춤) - 필요시 주석 해제
        # img = cv2.flip(img, 1)

        # 2. 전체 캔버스 만들기 (왼쪽: 지도, 오른쪽: 카메라)
        canvas = np.zeros((HEIGHT, TOTAL_WIDTH, 3), dtype=np.uint8)
        
        # 3. 수화 인식 로직
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(image_rgb)

        left_hand, right_hand = None, None
        if result.multi_hand_landmarks:
            for hand_lms, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                self.mp_drawing.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                label = handedness.classification[0].label
                if label == 'Left': left_hand = hand_lms
                else: right_hand = hand_lms
        
        # 데이터 수집
        self.seq_buffer.append(extract_xyz(left_hand) + extract_xyz(right_hand))

        # 예측
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

        # 4. 로봇 업데이트 및 지도 그리기
        self.update_robot()
        self.draw_map(canvas)

        # 5. 오른쪽 화면(카메라) 배치
        # 카메라 크기를 500x375 정도로 조정해서 배치
        img_resized = cv2.resize(img, (CAM_WIDTH, int(CAM_WIDTH * 0.75)))
        y_offset = (HEIGHT - img_resized.shape[0]) // 2
        canvas[y_offset:y_offset+img_resized.shape[0], MAP_WIDTH:TOTAL_WIDTH] = img_resized

        # 6. 정보 텍스트 (오른쪽 상단)
        info_x = MAP_WIDTH + 20
        cv2.putText(canvas, f"STATUS: {self.status}", (info_x, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(canvas, f"ACTION: {self.last_action.upper()}", (info_x, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(canvas, f"CONF: {self.confidence*100:.1f}%", (info_x, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 최종 영상을 Streamlit으로 반환
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

# 모델 로드 확인
model = load_model()
if model is None:
    st.error("❌ 모델 파일(model.joblib)을 찾을 수 없습니다. 같은 폴더에 넣어주세요.")
else:
    # WebRTC 스트리머 실행
    ctx = webrtc_streamer(
        key="sign-language",
        video_processor_factory=SignLanguageProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
