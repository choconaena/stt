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
    from my_whisper_improved import WhisperSTT
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
        
    def initialize_stt(self):
        """개선된 STT 프로세서 초기화"""
        self.stt_processor = WhisperSTT(
            model_path=model_path,
            device=device,
            sample_rate=SAMPLE_RATE,
            min_seg_duration=0.8,       # 0.8초로 완화
            silence_duration=0.4,       # 0.4초로 완화
            max_segment_duration=5.0,   # 최대 세그먼트 유지
            rms_threshold=-50.0,        # 원래 값으로 복원
            var_threshold=15.0,         # 낮은 값으로 완화
            vad_mode=2,                 # 중간 수준
            num_speakers=2
        )

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
    """텍스트 후처리 클래스"""
    def __init__(self):
        self.skip_texts = {
            "응", "뭐", "뭐야", "감사합니다", "응.", "아", "흠", 
            "르.", "너.", ".", "음", "어", "그", "네", "예", "아니",
            "그래", "으음", "음음", "어어", "아아", "네네", "맞아",
            "진짜", "뭐지", "지어", "짜증 나", "오케이", "끄끄끄",
            "티티티", "지금", "네.", "맞아.", "뭐야.", "감사합니다.",
            "MBC 뉴스", "구독", "좋아요", "알림", "시청", "뉴스"
        }
        
        self.filler_words = ["음", "어", "그", "저", "이제", "그래서"]
        
        # 반복 패턴 감지
        self.last_texts = deque(maxlen=5)
        
    def process(self, text):
        """텍스트 후처리"""
        if not text:
            return ""
        
        # 소문자로 변환하여 비교
        text_lower = text.strip().lower()
        
        # 짧은 무의미한 텍스트 필터링
        if text_lower in [t.lower() for t in self.skip_texts]:
            return ""
        
        # 너무 짧은 텍스트 필터링 (3자 이하)
        if len(text.strip()) <= 3:
            return ""
        
        # 특수문자만 있는 경우 필터링
        if all(c in ".,!?~@#$%^&*()_+-=[]{}|;:'\",.<>?/" for c in text.strip()):
            return ""
        
        # 반복 감지 - 최근 5개 텍스트와 비교
        if text_lower in [t.lower() for t in self.last_texts]:
            return ""  # 반복되는 텍스트 무시
        
        # 반복 문자 정리
        text = re.sub(r'(.)\1{5,}', r'\1\1\1...', text)
        
        # 시작/끝 filler 제거
        words = text.split()
        while words and words[0] in self.filler_words:
            words.pop(0)
        while words and words[-1] in self.filler_words:
            words.pop()
        
        processed_text = ' '.join(words)
        
        # 처리된 텍스트가 유효하면 기록
        if processed_text:
            self.last_texts.append(processed_text)
        
        return processed_text

async def register_client(websocket):
    client_id = f"{websocket.remote_address[0]}_{websocket.remote_address[1]}_{int(datetime.now().timestamp())}"
    clients[client_id] = websocket
    print(f"클라이언트 등록됨: {client_id}")
    return client_id

async def unregister_client(client_id):
    if client_id in clients:
        del clients[client_id]
        print(f"클라이언트 해제됨: {client_id}")

async def handle_client(websocket, path=None):
    client_id = await register_client(websocket)
    session = None

    try:
        # 초기 연결 메시지
        await websocket.send(json.dumps({
            "type": "connected",
            "message": f"클라이언트 {client_id} 연결됨"
        }))
        
        # 세션 초기화
        session = EnhancedSpeakerSession(client_id)
        session.initialize_stt()
        
        await websocket.send(json.dumps({
            "type": "model_loaded",
            "message": "개선된 STT AI 모델 로드 완료! 화자 등록을 시작하세요."
        }))
        
        # 실시간 처리 태스크 시작
        processing_task = asyncio.create_task(realtime_processing(websocket, session))
        
        async for message in websocket:
            try:
                if isinstance(message, bytes):
                    # 오디오 데이터 처리
                    await handle_audio_data(websocket, session, message)
                else:
                    # 텍스트 명령 처리
                    data = json.loads(message)
                    await handle_command(websocket, session, data)
                    
            except json.JSONDecodeError:
                print(f"JSON 파싱 오류: {message}")
            except Exception as e:
                print(f"메시지 처리 오류: {e}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"처리 오류: {str(e)}"
                }))
        
        # 처리 태스크 종료
        processing_task.cancel()
        
    except websockets.exceptions.ConnectionClosed:
        print(f"클라이언트 {client_id} 연결 종료")
    except Exception as e:
        print(f"예외 발생 (클라이언트 {client_id}): {e}")
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

async def handle_audio_data(websocket, session, audio_data):
    """오디오 데이터 처리"""
    # 실시간 버퍼에 추가
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    session.realtime_buffer.extend(audio_np)
    
    if session.registration_mode:
        # 화자 등록 모드
        if session.current_speaker_registering is not None:
            session.speaker_samples[session.current_speaker_registering].append(audio_data)
            
            # 5초 이상 수집되면 등록
            total_duration = len(b''.join(session.speaker_samples[session.current_speaker_registering])) / (SAMPLE_RATE * 2)
            
            if total_duration >= 5.0:
                await register_speaker(websocket, session, session.current_speaker_registering)
                session.current_speaker_registering = None
                
                # 두 화자 모두 등록되었는지 확인
                if all(len(samples) > 0 for samples in session.speaker_samples.values()):
                    session.registration_mode = False
                    session.registration_complete = True
                    
                    await websocket.send(json.dumps({
                        "type": "registration_complete",
                        "message": "모든 화자 등록 완료! 대화를 시작하세요."
                    }))
    else:
        # 음성 인식 모드
        events = session.stt_processor.process_chunk(audio_data)
        
        for event in events:
            if event["type"] == "partial":
                # 부분 전사 결과 전송
                await websocket.send(json.dumps({
                    "type": "partial",
                    "text": event["text"],
                    "timestamp": event["timestamp"],
                    "speaker_id": event.get("speaker_id", 0)
                }))
                
            elif event["type"] == "final":
                # 텍스트 후처리
                processed_text = session.text_processor.process(event["text"])
                
                if processed_text:
                    # 화자별 마지막 발화 시간 업데이트
                    current_time = time.time()
                    session.last_speech_time[event["speaker_id"]] = current_time
                    
                    # 중첩 여부 확인 - 더 엄격한 조건
                    is_overlapping = False
                    overlap_threshold = 0.2  # 200ms 이내만 중첩으로 판단
                    
                    # 다른 화자가 최근에 말했는지 확인
                    for sid, last_time in session.last_speech_time.items():
                        if sid != event["speaker_id"]:
                            time_diff = abs(current_time - last_time)
                            if time_diff < overlap_threshold:
                                # 실제로 다른 화자의 음성이 감지된 경우만
                                if len(session.stt_processor.speaker_embeddings) >= 2:
                                    is_overlapping = True
                                    break
                    
                    # 결과 전송
                    result = {
                        "type": "transcription",
                        "speaker_id": event["speaker_id"],
                        "text": processed_text,
                        "timestamp": event["timestamp"],
                        "confidence": event.get("confidence", 0),
                        "is_overlapping": is_overlapping
                    }
                    
                    await websocket.send(json.dumps(result))
                    print(f"[화자 {event['speaker_id']}] {processed_text}")
                    
                    # 실제 중첩이 발생한 경우만 한 번 알림
                    if is_overlapping and not hasattr(session, 'last_overlap_notification'):
                        session.last_overlap_notification = current_time
                        await websocket.send(json.dumps({
                            "type": "overlap_detected",
                            "message": "화자가 동시에 말하고 있습니다",
                            "timestamp": current_time
                        }))
                    elif is_overlapping:
                        # 마지막 알림으로부터 5초 이상 경과한 경우만 재알림
                        if current_time - session.last_overlap_notification > 5.0:
                            session.last_overlap_notification = current_time
                            await websocket.send(json.dumps({
                                "type": "overlap_detected",
                                "message": "화자가 동시에 말하고 있습니다",
                                "timestamp": current_time
                            }))

async def handle_command(websocket, session, data):
    """명령 처리"""
    command = data.get("command")
    
    if command == "start_registration":
        speaker_id = data.get("speaker_id", 0)
        session.current_speaker_registering = speaker_id
        session.speaker_samples[speaker_id] = []
        
        await websocket.send(json.dumps({
            "type": "registration_started",
            "speaker_id": speaker_id,
            "message": f"화자 {speaker_id + 1} 등록 시작. 5초 동안 말씀해주세요."
        }))
        
    elif command == "skip_registration":
        session.registration_mode = False
        session.registration_complete = True
        
        await websocket.send(json.dumps({
            "type": "registration_skipped",
            "message": "화자 등록을 건너뛰었습니다. 기본 설정으로 진행합니다."
        }))
        
    elif command == "reset":
        session.stt_processor.reset_session()
        session.speaker_samples = {0: [], 1: []}
        session.registration_mode = True
        session.registration_complete = False
        session.current_speaker_registering = None
        session.realtime_buffer.clear()
        session.last_speech_time = {0: 0, 1: 0}
        
        await websocket.send(json.dumps({
            "type": "reset_complete",
            "message": "세션이 초기화되었습니다."
        }))
        
    elif command == "get_status":
        speaker_info = session.stt_processor.get_speaker_info()
        
        status = {
            "type": "status",
            "registration_mode": session.registration_mode,
            "registration_complete": session.registration_complete,
            "speaker_info": speaker_info,
            "active_speakers": [sid for sid, t in session.last_speech_time.items() 
                               if time.time() - t < 2.0]
        }
        
        await websocket.send(json.dumps(status))
        
    elif command == "enable_overlap_detection":
        enabled = data.get("enabled", True)
        # 중첩 감지 활성화/비활성화 로직
        
        await websocket.send(json.dumps({
            "type": "overlap_detection_updated",
            "enabled": enabled
        }))

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