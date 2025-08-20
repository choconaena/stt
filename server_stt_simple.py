# -*- coding: utf-8 -*-
"""
실시간 한국어 STT 서버 (화자분리 포함, WSS) - STT 전용
- Whisper 기반 ASR + 화자분리
- 화자 등록 프로세스 포함
- WebSocket Secure 서버: wss://0.0.0.0:8085
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

from my_whisper import WhisperSTT

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
HOST = "0.0.0.0"
PORT = 8088

model_path = "/home/2020112534/safe_hi/model/my_whisper"

# TLS 설정 - 로컬 key 디렉토리의 인증서 사용
CERT_FILE = "./key/fullchain.pem"
KEY_FILE = "./key/privkey.pem"

ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_ctx.load_cert_chain(certfile=CERT_FILE, keyfile=KEY_FILE)
ssl_ctx.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1

# 전역 클라이언트 관리용
clients = {}

class SpeakerSession:
    def __init__(self, client_id):
        self.client_id = client_id
        self.stt_processor = None
        self.registration_mode = True
        self.current_speaker_registering = None
        self.speaker_samples = {0: [], 1: []}  # 각 화자별 샘플 저장
        self.registration_complete = False
        
    def initialize_stt(self):
        self.stt_processor = WhisperSTT(
            model_path=model_path,
            device=device,
            sample_rate=SAMPLE_RATE,
            partial_interval=1.0,
            min_seg_duration=1.0,
            silence_duration=0.5,
            rms_threshold=-50.0,
            var_threshold=20.0,
            vad_mode=2,
            num_speakers=2
        )

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
        # 1) 초기 연결 메시지
        await websocket.send(json.dumps({
            "type": "connected",
            "message": f"클라이언트 {client_id} 연결됨"
        }))
        
        # 세션 초기화
        session = SpeakerSession(client_id)
        session.initialize_stt()
        
        await websocket.send(json.dumps({
            "type": "model_loaded",
            "message": "STT AI 모델 로드 완료! 화자 등록을 시작하세요."
        }))
        
        async for message in websocket:
            try:
                # JSON 메시지인지 확인
                if isinstance(message, str):
                    data = json.loads(message)
                    await handle_control_message(websocket, session, data)
                else:
                    # 바이너리 오디오 데이터
                    await handle_audio_data(websocket, session, message)
                    
            except json.JSONDecodeError:
                # 바이너리 오디오 데이터로 처리
                await handle_audio_data(websocket, session, message)
        
    except websockets.exceptions.ConnectionClosed:
        print(f"클라이언트 {client_id} 접속 종료됨.")
    except Exception as e:
        print(f"[{client_id}] 오류 발생: {e}")
    finally:
        await unregister_client(client_id)

async def handle_control_message(websocket, session, data):
    msg_type = data.get("type")
    
    if msg_type == "start_speaker_registration":
        speaker_id = data.get("speaker_id")
        session.current_speaker_registering = speaker_id
        session.registration_mode = True
        
        await websocket.send(json.dumps({
            "type": "speaker_registration_started",
            "speaker_id": speaker_id,
            "message": f"화자 {speaker_id + 1}번의 목소리를 등록해주세요. (3-5초간 말씀해주세요)"
        }))
    
    elif msg_type == "complete_speaker_registration":
        speaker_id = session.current_speaker_registering
        if speaker_id is not None and len(session.speaker_samples[speaker_id]) > 0:
            # 화자 등록 완료 처리
            await register_speaker_samples(session, speaker_id)
            session.current_speaker_registering = None
            
            await websocket.send(json.dumps({
                "type": "speaker_registration_completed",
                "speaker_id": speaker_id,
                "message": f"화자 {speaker_id + 1}번 등록이 완료되었습니다."
            }))
    
    elif msg_type == "start_transcription":
        session.registration_mode = False
        session.registration_complete = True
        
        await websocket.send(json.dumps({
            "type": "transcription_started",
            "message": "실시간 전사를 시작합니다."
        }))

async def handle_audio_data(websocket, session, audio_chunk):
    if not session.stt_processor:
        return
    
    if session.registration_mode and session.current_speaker_registering is not None:
        # 화자 등록 모드
        speaker_id = session.current_speaker_registering
        session.speaker_samples[speaker_id].append(audio_chunk)
        
    elif session.registration_complete:
        # 실시간 전사 모드
        events = session.stt_processor.process_chunk(audio_chunk)
        for evt in events:
            text = evt["text"]
            
            if "[무음]" in text:
                continue
            
            # 화자 정보 추출
            speaker_match = re.search(r'\[화자 (\d+)\]:\s*(.*)', text)
            if speaker_match:
                speaker_id = int(speaker_match.group(1))
                cleaned_text = speaker_match.group(2).strip()
            else:
                # 기본 형식 처리
                cleaned_text = re.sub(r"\[\d+번문장\]:\s*", "", text)
                speaker_id = 0  # 기본 화자
            
            # 같은 글자 반복 처리
            cleaned_text = re.sub(r'(.)\1{5,}', r'\1\1\1\1\1\1 ...', cleaned_text)
            
            if cleaned_text.strip():
                print(f"[SEND to {session.client_id}] Speaker {speaker_id}: {cleaned_text}")
                
                # 클라이언트로 전송
                await websocket.send(json.dumps({
                    "type": "transcription",
                    "speaker_id": speaker_id,
                    "text": cleaned_text,
                    "timestamp": datetime.now().isoformat()
                }))

async def register_speaker_samples(session, speaker_id):
    """화자 샘플을 STT 프로세서에 등록"""
    if len(session.speaker_samples[speaker_id]) == 0:
        return
    
    # 모든 샘플을 하나로 합치기
    combined_audio = bytearray()
    for sample in session.speaker_samples[speaker_id]:
        combined_audio.extend(sample)
    
    # numpy 배열로 변환
    audio_np = np.frombuffer(combined_audio, dtype=np.int16).astype(np.float32) / 32768.0
    
    # STT 프로세서에 화자 등록
    if len(audio_np) > SAMPLE_RATE * 0.5:  # 최소 0.5초 이상
        embedding = session.stt_processor.extract_embedding(audio_np)
        if embedding is not None:
            session.stt_processor.speaker_embeddings.append((embedding, speaker_id))
            session.stt_processor.initial_enroll_count += 1
            print(f"화자 {speaker_id} 등록 완료 (샘플 길이: {len(audio_np)/SAMPLE_RATE:.2f}초)")

async def main():
    # SSL 인증서 파일 확인
    if not os.path.exists(CERT_FILE):
        print(f"❌ SSL 인증서를 찾을 수 없습니다: {CERT_FILE}")
        return
    if not os.path.exists(KEY_FILE):
        print(f"❌ SSL 키 파일을 찾을 수 없습니다: {KEY_FILE}")
        return
    
    print(f"✅ SSL 인증서 확인됨: {CERT_FILE}")
    print(f"✅ SSL 키 파일 확인됨: {KEY_FILE}")
    
    try:
        ssl_ctx.load_cert_chain(certfile=CERT_FILE, keyfile=KEY_FILE)
        print("✅ SSL 컨텍스트 로드 완료")
    except Exception as e:
        print(f"❌ SSL 설정 오류: {e}")
        return
    
    try:
        async with websockets.serve(
            handle_client,
            HOST,
            PORT,
            ssl=ssl_ctx,
            max_size=None,
            compression=None
        ):
            print(f"🚀 STT 서버가 wss://{HOST}:{PORT} 대기중...")
            print(f"🌐 외부 접속: wss://safe-hi.xyz:{PORT}")
            print("Ctrl+C로 종료")
            await asyncio.Future()
    except Exception as e:
        print(f"❌ 서버 시작 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())