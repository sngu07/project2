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
MODEL_PATH = 'seq_sign_model_final.joblib' 
SEQ_LENGTH = 40
MAP_WIDTH = 500
CAM_WIDTH = 500
HEIGHT = 700
TOTAL_WIDTH = MAP_WIDTH + CAM_WIDTH
BG_COLOR = (40, 40, 40)
ROOM_COLOR = (200, 200, 200)
ROBOT_COLOR = (50, 50, 255)
TEXT_COLOR = (0, 0, 0)

ROOMS = {
    'toilet':   (250, 100, 200, 80),
    'room2':    (250, 220, 200, 80),
    'room1':    (250, 340, 200, 80),
    'elevator': (250, 460, 200, 80),
    'home':     (250, 600, 80, 50)
}

# =========================================================
# 2. 모델 로드 (캐싱)
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
        cv2.rectangle(canvas, (0, 0), (MAP_WIDTH, HEIGHT), BG_COLOR, -1)
        cv2.line(canvas, (250, 100), (250, 600), (100, 100, 100), 10)
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
        cv2.circle(canvas, (int(self.rx), int(self.ry)), 15, ROBOT_COLOR, -1)
        cv2.circle(canvas, (int(self.rx), int(self.ry)), 15, (255,255,255), 2)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
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

        self.update_robot()
        self.draw_map(canvas)
        img_resized = cv2.resize(img, (CAM_WIDTH, int(CAM_WIDTH * 0.75)))
        y_offset = (HEIGHT - img_resized.shape[0]) // 2
        canvas[y_offset:y_offset+img_resized.shape[0], MAP_WIDTH:TOTAL_WIDTH] = img_resized
        
        info_x = MAP_WIDTH + 20
        cv2.putText(canvas, f"STATUS: {self.status}", (info_x, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(canvas, f"ACTION: {self.last_action.upper()}", (info_x, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(canvas, f"CONF: {self.confidence*100:.1f}%", (info_x, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return av.VideoFrame.from_ndarray(canvas, format="bgr24")

# =========================================================
# 5. 페이지 함수 정의
# =========================================================

def show_intro_page():
    st.title("📘 수화 동작 가이드")
    st.markdown("""
    아래의 수화 동작을 웹캠에 보여주면 로봇이 인식하여 해당 장소로 이동하거나 복귀합니다.  
    **[동작 영상 보기]** 버튼을 누르면 국립국어원 한국수어사전의 정확한 영상으로 연결됩니다.
    """)
    st.divider()

    # 3개의 컬럼으로 나누어 배치
    col1, col2, col3 = st.columns(3)

    # 1. 화장실 (Toilet)
    with col1:
        st.subheader("🚽 화장실 (Toilet)")
        st.markdown("**명령: `Move to Toilet`**")
        st.info("로봇이 **화장실** 구역으로 이동합니다.")
        st.write("오른 주먹의 1·5지를 펴서 코를 쥐었다가 떼며 주먹을 쥡니다.")
        st.link_button("▶️ 동작 영상 보기", "https://sldict.korean.go.kr/front/sign/signContentsView.do?origin_no=971&top_category=CTE&category=&searchKeyword=%ED%99%94%EC%9E%A5%EC%8B%A4&searchCondition=&search_gubun=&museum_type=00&current_pos_index=0")

    # 2. 강의실 (Lecture Room) -> Room1/2 매핑
    with col2:
        st.subheader("🏫 강의실 (Classroom)")
        st.markdown("**명령: `Move to Room`**")
        st.info("로봇이 **Room 1** 또는 **Room 2**로 이동합니다.")
        st.write("두 주먹을 쥐고 손목을 엇걸어 두 번 두드립니다.") # 수어사전 설명 요약
        st.link_button("▶️ 동작 영상 보기", "https://sldict.korean.go.kr/front/sign/signContentsView.do?origin_no=6305&top_category=CTE&category=&searchKeyword=%EA%B5%90%EC%8B%A4&searchCondition=&search_gubun=&museum_type=00&current_pos_index=0")

    # 3. 고마워 (Thank You) -> Home 복귀
    with col3:
        st.subheader("🙇 고마워 (Thanks)")
        st.markdown("**명령: `Return Home`**")
        st.success("로봇이 **시작 지점(Home)**으로 복귀합니다.")
        st.write("손을 펴서 손날로 다른 손의 손등을 두 번 두드립니다.")
        st.link_button("▶️ 동작 영상 보기", "https://sldict.korean.go.kr/front/sign/signContentsView.do?origin_no=2372&top_category=CTE&category=&searchKeyword=%EA%B0%90%EC%82%AC&searchCondition=&search_gubun=&museum_type=00&current_pos_index=0")

    st.divider()
    st.warning("⚠️ **Tip**: 웹캠 정면에서 손 동작을 크고 정확하게 해주세요.")

def show_simulation_page():
    st.header("🤖 Robot Simulation")
    
    # 1. 모델 로딩 대기 표시 (사용자가 멈춘 줄 알지 않게 함)
    with st.spinner("🧠 AI 모델을 불러오는 중입니다... (최초 1회는 시간이 걸릴 수 있습니다)"):
        if not os.path.exists(MODEL_PATH):
            st.error(f"⚠️ `{MODEL_PATH}` 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
            return
        # 모델을 미리 로드하여 캐싱 확실히 하기
        load_model()

    st.markdown("""
    왼쪽은 **로봇 시뮬레이션**, 오른쪽은 **웹캠**입니다.  
    카메라가 켜질 때까지 **5~10초** 정도 걸릴 수 있습니다. 'START' 버튼을 눌러주세요.
    """)

    # 2. 연결 속도 개선을 위한 STUN 서버 추가
    # 구글의 기본 서버 외에 백업 서버들을 추가하여 연결 성공률을 높입니다.
    rtc_config = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
        ]
    })

    # 3. WebRTC 스트리머 실행
    ctx = webrtc_streamer(
        key="sign-language",
        video_processor_factory=SignLanguageProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 480},  # 해상도를 낮춰서 전송 속도 향상
                "height": {"ideal": 360}, 
                "frameRate": {"ideal": 15} # 프레임 수를 낮춰서 버벅임 방지
            }, 
            "audio": False
        },
        async_processing=True,
    )
# =========================================================
# 6. Main App Structure
# =========================================================
def main():
    st.set_page_config(page_title="AI Robot Navigation", layout="wide")
    
    # 사이드바에서 페이지 선택
    st.sidebar.title("메뉴")
    page = st.sidebar.radio("이동할 페이지를 선택하세요:", ["수화 가이드", "로봇 시뮬레이션"])

    if page == "수화 가이드":
        show_intro_page()
    elif page == "로봇 시뮬레이션":
        show_simulation_page()

if __name__ == "__main__":
    main()

