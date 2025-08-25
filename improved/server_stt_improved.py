# server_stt_improved.py
# -*- coding: utf-8 -*-
"""
실시간 한국어 STT 서버 (개선된 화자분리) - 중첩 대화 지원
- 슬라이딩 윈도우 기반 실시간 처리
- 다중 화자 동시 감지
- 개선된 화자 추적 및 스무딩
"""

import asyncio
import websockets
import os
from datetime import datetime
import torch
import json
import re
import ssl
import numpy as np
from collections import deque
import time
import sys

# 개선된 my_whisper 모듈 임포트 - 경로 수정 필요할 수 있음
try:
    from my_whisper_improved import ImprovedWhisperSTT
except ImportError:
    print("my_whisper_improved.py 파일을 찾을 수 없습니다.")
    print("같은 디렉토리에 my_whisper_improved.py 파일이 있는지 확인하세요.")
    sys.exit(1)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
HOST = "0.0.0.0"
PORT = 8088

model_path = "/home/2020112534/safe_hi/model/my_whisper"

# TLS 설정
CERT_FILE = "./key/fullchain.pem"
KEY_FILE = "./key/privkey.pem"

ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_ctx.load_cert_chain(certfile=CERT_FILE, keyfile=KEY_FILE)
# SSL 옵션 설정 (더 이상 사용되지 않는 옵션 제거)
ssl_ctx.minimum_version = ssl.TLSVersion.TLSv1_2  # TLS 1.2 이상만 허용

# 전역 클라이언트 관리
clients = {}

class EnhancedSpeakerSession:
    def __init__(self, client_id):
        self.client_id = client_id
        self.stt_processor = None
        self.registration_mode = True
        self.current_speaker_registering = None
        self.speaker_samples = {0: [], 1: []}
        self.registration_complete = False
        
        # 실시간 처리를 위한 버퍼
        self.realtime_buffer = deque(maxlen=int(SAMPLE_RATE * 0.5))  # 0.5초 버퍼
        self.last_process_time = time.time()
        
        # 화자별 마지막 발화 시간 추적
        self.last_speech_time = {0: 0, 1: 0}
        
        # 중첩 감지용
        self.overlap_detector = OverlapDetector()
        
        # 텍스트 후처리용
        self.text_processor = TextPostProcessor()

class OverlapDetector:
    """화자 중첩 감지 클래스"""
    def __init__(self, window_size=0.5, energy_threshold=-40):
        self.window_size = window_size
        self.energy_threshold = energy_threshold
        self.speaker_energies = {0: deque(maxlen=10), 1: deque(maxlen=10)}
        self.last_overlap_time = 0
        self.overlap_cooldown = 2.0  # 2초 쿨다운
        
    def detect_overlap(self, audio_np, speaker_embeddings, embedding_model):
        """오디오에서 화자 중첩 감지 - 실제 중첩만 감지"""
        # 실제로 2명 이상의 화자가 등록되어 있을 때만 작동
        if len(speaker_embeddings) < 2:
            return False, []
        
        # 쿨다운 체크 - 너무 자주 알림 방지
        current_time = time.time()
        if current_time - self.last_overlap_time < self.overlap_cooldown:
            return False, []
        
        # 여기서는 실제 중첩 감지를 비활성화
        # 실제 중첩은 화자별 마지막 발화 시간으로 판단
        return False, []

class TextPostProcessor:
    """강화된 텍스트 후처리 클래스"""
    def __init__(self):
        # 무시할 텍스트들 (더 많이 추가)
        self.skip_texts = {
            # 짧은 감탄사
            "응", "음", "어", "아", "오", "우", "에", "이", "으", "흠", "헉", "어어", "아아", "음음",
            # 의미없는 짧은 말들
            "뭐", "뭐야", "뭐지", "그", "네", "예", "아니", "맞아", "그래", "진짜", "어?", "뭐?",
            # 문장부호나 특수문자만
            ".", ",", "?", "!", "~", "-", "=", "+", "*", "(", ")", "[", "]", "{", "}",
            # 의미없는 반복
            "르", "너", "지어", "끄끄끄", "티티티", "지지지", "끼방", "응.", "네.", "아.", "음.", "어.",
            # 뉴스/방송 관련
            "MBC 뉴스", "구독", "좋아요", "알림", "시청", "뉴스", "방송", "채널", "구독자",
            # 기타 불필요한 것들
            "감사합니다", "안녕하세요", "안녕히", "수고", "오케이", "okay", "ok"
        }
        
        # 필러 워드 (앞뒤 제거용)
        self.filler_words = ["음", "어", "그", "저", "이제", "그래서", "뭐", "응", "네", "예"]
        
        # 최근 텍스트 기록 (중복 방지)
        self.last_texts = deque(maxlen=10)
        
        # 반복 패턴 감지를 위한 설정
        self.min_meaningful_length = 6  # 최소 의미있는 길이
        self.max_repetition_ratio = 0.7  # 반복 비율 임계값
        
    def process(self, text):
        """강화된 텍스트 후처리"""
        if not text or not isinstance(text, str):
            return ""
        
        # 1. 기본 정리
        text = text.strip()
        if not text:
            return ""
        
        # 2. 길이 체크 - 너무 짧은 것은 제외
        if len(text) <= 2:
            return ""
        
        # 3. 소문자 변환해서 스킵 리스트와 비교
        text_lower = text.lower()
        if text_lower in [t.lower() for t in self.skip_texts]:
            return ""
        
        # 4. 특수문자만 있는 경우 제외
        if all(c in ".,!?~@#$%^&*()_+-=[]{}|;:'\",.<>/\\" for c in text):
            return ""
        
        # 5. 숫자만 있는 경우 제외
        if text.replace(" ", "").isdigit():
            return ""
        
        # 6. 과도한 반복 문자 체크 (예: "아아아아아..." -> "아...")
        text = re.sub(r'(.)\1{4,}', r'\1\1\1', text)  # 4번 이상 반복되면 3번으로 축약
        
        # 7. 반복되는 단어 체크 (예: "나나나나나..." -> 제거)
        if self._is_excessive_repetition(text):
            return ""
        
        # 8. 최근 텍스트와 중복 체크
        if text_lower in [t.lower() for t in self.last_texts]:
            return ""
        
        # 9. 의미있는 길이 체크
        meaningful_chars = re.sub(r'[^\w가-힣]', '', text)  # 한글, 영문, 숫자만 추출
        if len(meaningful_chars) < 3:  # 의미있는 문자가 3자 미만이면 제외
            return ""
        
        # 10. filler word 제거
        words = text.split()
        # 앞쪽 filler 제거
        while words and words[0].lower() in [f.lower() for f in self.filler_words]:
            words.pop(0)
        # 뒤쪽 filler 제거
        while words and words[-1].lower() in [f.lower() for f in self.filler_words]:
            words.pop()
        
        processed_text = ' '.join(words).strip()
        
        # 11. 최종 길이 체크
        if len(processed_text) < 4:  # 처리 후 4자 미만이면 제외
            return ""
        
        # 12. 유효한 텍스트면 기록
        if processed_text:
            self.last_texts.append(processed_text)
        
        return processed_text
    
    def _is_excessive_repetition(self, text):
        """과도한 반복 패턴 감지"""
        if len(text) < 6:
            return False
        
        # 1-3글자 단위로 반복 체크
        for unit_length in range(1, 4):
            if len(text) < unit_length * 3:  # 최소 3번은 반복되어야 함
                continue
            
            unit = text[:unit_length]
            repeated_text = unit * (len(text) // unit_length + 1)
            
            # 원본 텍스트와 반복 텍스트의 유사도 계산
            similarity = self._calculate_similarity(text, repeated_text[:len(text)])
            
            if similarity > self.max_repetition_ratio:
                return True
        
        return False
    
    def _calculate_similarity(self, text1, text2):
        """두 텍스트의 유사도 계산 (간단한 방식)"""
        if not text1 or not text2:
            return 0.0
        
        matches = sum(1 for c1, c2 in zip(text1, text2) if c1 == c2)
        return matches / max(len(text1), len(text2))

async def register_client(websocket):
    client_id = f"{websocket.remote_address[0]}_{websocket.remote_address[1]}_{int(datetime.now().timestamp())}"
    clients[client_id] = websocket
    print(f"클라이언트 등록됨: {client_id}")
    return client_id

async def unregister_client(client_id):
    if client_id in clients:
        del clients[client_id]
        print(f"클라이언트 해제됨: {client_id}")

# 메인 핸들러 수정
async def handle_client(websocket, path=None):
    """개선된 클라이언트 핸들러"""
    client_id = await register_client(websocket)
    session = None

    try:
        # 초기 연결 메시지
        await websocket.send(json.dumps({
            "type": "connected",
            "message": f"클라이언트 {client_id} 연결됨"
        }))
        
        # 세션 초기화 - 개선된 STT 사용
        session = EnhancedSpeakerSession(client_id)
        session.stt_processor = ImprovedWhisperSTT(
            model_path=model_path,
            device=device,
            sample_rate=SAMPLE_RATE,
            num_speakers=2
        )
        
        await websocket.send(json.dumps({
            "type": "model_loaded",
            "message": "STT AI 모델 로드 완료! 화자 등록을 시작하세요."
        }))
        
        # 메시지 처리 루프
        async for message in websocket:
            try:
                if isinstance(message, bytes):
                    # 오디오 데이터 처리
                    await improved_handle_audio_data(websocket, session, message)
                else:
                    # 텍스트 명령 처리
                    data = json.loads(message)
                    await improved_handle_command(websocket, session, data)
                    
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류: {e}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "잘못된 메시지 형식입니다."
                }))
            except Exception as e:
                print(f"메시지 처리 오류: {e}")
                import traceback
                traceback.print_exc()
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"처리 오류: {str(e)}"
                }))
        
    except websockets.exceptions.ConnectionClosed:
        print(f"클라이언트 {client_id} 연결 종료")
    except Exception as e:
        print(f"예외 발생 (클라이언트 {client_id}): {e}")
        import traceback
        traceback.print_exc()
    finally:
        if session and session.stt_processor:
            session.stt_processor.cleanup()
        await unregister_client(client_id)


async def realtime_processing(websocket, session):
    """실시간 처리 루프 - 중첩 감지 제거"""
    while True:
        try:
            # 100ms마다 체크
            await asyncio.sleep(0.1)
            
            # 버퍼에 데이터가 있으면 처리
            if len(session.realtime_buffer) > 0:
                now = time.time()
                
                # 마지막 처리 후 0.3초 이상 경과했으면 처리
                if now - session.last_process_time > 0.3:
                    session.last_process_time = now
                    # 중첩 감지 부분 제거 - 실제 화자 교체 시에만 판단
                    
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"실시간 처리 오류: {e}")


# 오디오 처리 부분도 수정
async def improved_handle_audio_data(websocket, session, audio_data):
    """개선된 오디오 데이터 처리 - 더 엄격한 필터링"""
    
    if session.registration_mode:
        # 화자 등록 모드 (기존 코드 유지)
        if session.current_speaker_registering is not None:
            session.speaker_samples[session.current_speaker_registering].append(audio_data)
            
            total_duration = len(b''.join(session.speaker_samples[session.current_speaker_registering])) / (SAMPLE_RATE * 2)
            
            if total_duration >= 5.0:
                await register_speaker(websocket, session, session.current_speaker_registering)
                session.current_speaker_registering = None
                
                if all(len(samples) > 0 for samples in session.speaker_samples.values()):
                    session.registration_mode = False
                    session.registration_complete = True
                    
                    await websocket.send(json.dumps({
                        "type": "registration_complete",
                        "message": "모든 화자 등록 완료! 대화를 시작하세요."
                    }))
    else:
        # 음성 인식 모드 - 더 엄격한 필터링
        events = session.stt_processor.process_chunk(audio_data)
        
        for event in events:
            if not event.get("text"):
                continue
            
            # 원본 텍스트 길이 체크 - 너무 짧으면 아예 무시
            original_text = event["text"]
            if len(original_text.strip()) <= 3:
                continue
            
            # 부분 전사의 경우 더 관대하게, 최종 전사는 엄격하게
            if event["type"] == "partial":
                # 부분 전사는 길이만 체크하고 그대로 전송
                if len(original_text.strip()) >= 4:
                    await websocket.send(json.dumps({
                        "type": "partial",
                        "text": original_text,
                        "speaker_id": event.get("speaker_id", 0),
                        "segment_id": event.get("segment_id", 0),
                        "timestamp": event.get("timestamp", time.time())
                    }))
                        
            elif event["type"] == "final":
                # 최종 전사는 엄격한 후처리 적용
                processed_text = session.text_processor.process(original_text)
                
                # 후처리 후 텍스트가 있을 때만 전송
                if processed_text and len(processed_text) >= 4:
                    current_time = event.get("timestamp", time.time())
                    speaker_id = event.get("speaker_id", 0)
                    
                    # 화자별 마지막 발화 시간 업데이트
                    session.last_speech_time[speaker_id] = current_time
                    
                    # 중첩 체크 (기존 로직 유지)
                    is_overlapping = False
                    for sid, last_time in session.last_speech_time.items():
                        if sid != speaker_id and abs(current_time - last_time) < 0.5:
                            is_overlapping = True
                            break
                    
                    # 최종 결과 전송
                    await websocket.send(json.dumps({
                        "type": "transcription",
                        "speaker_id": speaker_id,
                        "text": processed_text,
                        "segment_id": event.get("segment_id", 0),
                        "timestamp": current_time,
                        "confidence": event.get("confidence", 0),
                        "is_overlapping": is_overlapping
                    }))
                    
                    print(f"[화자 {speaker_id}] {processed_text}")
                else:
                    # 필터링된 텍스트는 로그만 출력
                    print(f"[필터됨] 원본: '{original_text}' -> 후처리: '{processed_text}'")

async def improved_handle_command(websocket, session, data):
    """개선된 명령 처리"""
    command = data.get("type") or data.get("command")
    
    if command == "start_speaker_registration":
        speaker_id = data.get("speaker_id", 0)
        session.current_speaker_registering = speaker_id
        session.speaker_samples[speaker_id] = []
        
        await websocket.send(json.dumps({
            "type": "speaker_registration_started",
            "speaker_id": speaker_id,
            "message": f"화자 {speaker_id + 1}번의 목소리를 등록해주세요. (3-5초간 말씀해주세요)"
        }))
        
    elif command == "complete_speaker_registration":
        speaker_id = data.get("speaker_id", 0)
        if session.speaker_samples[speaker_id]:
            await register_speaker(websocket, session, speaker_id)
    
    elif command == "skip_registration":
        # 등록 건너뛰기 처리 추가
        session.registration_mode = False
        session.registration_complete = True
        session.current_speaker_registering = None
        
        await websocket.send(json.dumps({
            "type": "registration_skipped",
            "message": "화자 등록을 건너뛰었습니다. 기본 설정으로 진행합니다."
        }))
        
    elif command == "start_transcription":
        session.registration_mode = False
        await websocket.send(json.dumps({
            "type": "transcription_started",
            "message": "실시간 전사를 시작합니다."
        }))
        
    elif command == "reset":
        # 세션 리셋
        session.stt_processor.reset_session()
        session.speaker_samples = {0: [], 1: []}
        session.registration_mode = True
        session.registration_complete = False
        session.current_speaker_registering = None
        session.last_speech_time = {0: 0, 1: 0}
        
        # 텍스트 프로세서도 리셋
        session.text_processor = TextPostProcessor()
        
        await websocket.send(json.dumps({
            "type": "reset_complete",
            "message": "세션이 초기화되었습니다."
        }))
        
    elif command == "get_status":
        # 상태 정보 전송
        status = {
            "type": "status",
            "registration_mode": session.registration_mode,
            "registration_complete": session.registration_complete,
            "registered_speakers": len(session.stt_processor.speaker_embeddings) if hasattr(session.stt_processor, 'speaker_embeddings') else 0,
            "active_speakers": [
                sid for sid, t in session.last_speech_time.items() 
                if time.time() - t < 2.0
            ]
        }
        await websocket.send(json.dumps(status))


async def register_speaker(websocket, session, speaker_id):
    """화자 등록"""
    if not session.speaker_samples[speaker_id]:
        return
    
    # 모든 샘플을 하나로 결합
    all_audio = b''.join(session.speaker_samples[speaker_id])
    audio_np = np.frombuffer(all_audio, dtype=np.int16).astype(np.float32) / 32768.0
    
    # 화자 임베딩 등록
    session.stt_processor.register_speaker_embedding(audio_np, speaker_id)
    
    await websocket.send(json.dumps({
        "type": "speaker_registered",
        "speaker_id": speaker_id,
        "message": f"화자 {speaker_id + 1} 등록 완료!"
    }))
    
    print(f"화자 {speaker_id} 등록 완료 (샘플 길이: {len(audio_np) / SAMPLE_RATE:.1f}초)")

async def main():
    print(f"개선된 STT 서버 시작 - wss://{HOST}:{PORT}")
    print(f"디바이스: {device}")
    print("기능: 실시간 화자분리, 중첩 감지, 스무딩")
    
    async with websockets.serve(
        handle_client, 
        HOST, 
        PORT,
        ssl=ssl_ctx,
        max_size=10 * 1024 * 1024,  # 10MB
        ping_interval=20,
        ping_timeout=10
    ):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
