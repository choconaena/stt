# 🎤 실시간 화자분리 STT 시스템

두 명의 화자를 구분하여 실시간으로 음성을 텍스트로 변환하는 WebSocket 기반 STT 서비스입니다.

## ✨ 주요 기능

- **🎯 2명 화자 구분**: 음성 임베딩 기반 화자 식별
- **⚡ 실시간 처리**: WebSocket을 통한 저지연 스트리밍
- **🎨 직관적 UI**: 카카오톡 스타일 채팅 인터페이스
- **🔒 보안 통신**: HTTPS/WSS 암호화 통신
- **🌐 외부 접속**: SSH 터널링을 통한 원격 접속

## 🏗️ 시스템 아키텍처

```
브라우저 (HTTPS) → 네이버클라우드 (safe-hi.xyz) → SSH터널링 → 로컬서버
│                   │                              │
├─ 웹페이지 (8444)   ├─ 웹페이지 프록시              ├─ HTTPS 서버 (8444)
└─ WebSocket (8088) └─ WebSocket 프록시            └─ STT 서버 (8088)
```

## 📋 요구사항

### 하드웨어
- **GPU**: CUDA 지원 GPU (Whisper 모델용)
- **RAM**: 최소 8GB (모델 로딩용)
- **저장공간**: 최소 5GB (Whisper 모델 + 의존성)

### 소프트웨어
- **Python**: 3.11+
- **CUDA**: 11.8+ (GPU 사용 시)
- **OpenSSL**: SSL 인증서용
- **SSH**: 터널링용

### Python 의존성
```
torch>=2.0.0
transformers>=4.30.0
websockets>=11.0
librosa>=0.10.0
webrtcvad>=2.0.10
pyannote.audio>=3.1.0
scikit-learn>=1.3.0
numpy>=1.24.0
aiohttp>=3.8.0
```

## 🚀 설치 및 실행

### 1. 환경 설정
```bash
# 저장소 클론
git clone <repository-url>
cd stt

# Python 환경 생성 (conda 권장)
conda create -n stt_env python=3.11
conda activate stt_env

# 의존성 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers websockets librosa webrtcvad pyannote.audio scikit-learn numpy aiohttp
```

### 2. 모델 준비
```bash
# Whisper 모델 다운로드 (자동으로 캐시됨)
# 또는 커스텀 모델 경로 설정
# model_path = "/path/to/your/custom/whisper/model"
```

### 3. SSL 인증서 설정
```bash
# key 디렉토리에 SSL 인증서 배치
mkdir -p key/
# fullchain.pem과 privkey.pem 파일을 key/ 디렉토리에 복사
```

### 4. 서버 실행
```bash
# 터미널 1: STT 서버 실행
python server_stt_simple.py
python server_stt_improved.py

# 터미널 2: 웹서버 실행
python https_server.py


# 터미널 3: SSH 터널링 (원격 접속용)
ssh -N -R 0.0.0.0:8088:127.0.0.1:8088 -R 0.0.0.0:8444:127.0.0.1:8444 root@your-domain.com

ssh -N -R 0.0.0.0:8088:127.0.0.1:8088 -R 0.0.0.0:8444:127.0.0.1:8444 root@211.188.56.255
```

### 5. 브라우저 접속
```
https://your-domain.com:8444/speaker_stt_frontend.html

https://safe-hi.xyz:8444/speaker_stt_frontend.html
```

## 📖 사용 방법

### 1단계: 화자 등록
1. **화자 1 등록**: "목소리 등록하기" 버튼 클릭 → 3-5초간 말하기
2. **화자 2 등록**: 마찬가지로 두 번째 화자 등록
3. 등록 완료 시 ✅ 표시

### 2단계: 실시간 대화 전사
1. **"대화 전사 시작하기"** 버튼 클릭
2. 마이크 권한 허용
3. 실시간으로 화자별 구분되어 채팅창에 표시

### 3단계: 전사 중단
- **"전사 중단"** 버튼 클릭 또는 **ESC** 키

## ⚙️ 설정 가능한 옵션

### STT 파라미터 조정 (my_whisper.py)
```python
WhisperSTT(
    model_path="/path/to/model",        # Whisper 모델 경로
    device="cuda",                      # 'cuda' 또는 'cpu'
    sample_rate=16000,                  # 오디오 샘플레이트
    min_seg_duration=1.0,               # 최소 세그먼트 길이 (초)
    silence_duration=0.5,               # 무음 구간 길이 (초)
    max_segment_duration=7.0,           # 최대 세그먼트 길이 (초)
    rms_threshold=-50.0,                # 음성 감지 임계값 (dB)
    var_threshold=20.0,                 # 음성 특징 임계값
    vad_mode=2,                         # VAD 민감도 (0-3)
    num_speakers=2,                     # 화자 수
    similarity_threshold=0.6            # 화자 유사도 임계값
)
```

### 서버 포트 변경 (server_stt_simple.py)
```python
HOST = "0.0.0.0"    # 바인딩 주소
PORT = 8088         # WebSocket 포트
```

### 웹서버 포트 변경 (https_server.py)
```python
PORT = 8444         # HTTPS 웹서버 포트
```

## 🔧 문제 해결

### 일반적인 오류

#### 1. 마이크 접근 권한 오류
```
해결: 브라우저에서 마이크 권한 허용
Chrome: 주소창 왼쪽 🔒 아이콘 → 마이크 허용
```

#### 2. WebSocket 연결 실패
```bash
# 포트 충돌 확인
netstat -tlnp | grep 8088

# SSL 인증서 확인
openssl x509 -in ./key/fullchain.pem -text -noout

# 방화벽 설정
sudo ufw allow 8088
sudo ufw allow 8444
```

#### 3. CUDA 메모리 부족
```python
# GPU 번호 조정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 또는 CPU 모드 사용
device = "cpu"
```

#### 4. 화자분리 정확도 낮음
```python
# 등록 시간 늘리기 (3-5초 → 5-10초)
# similarity_threshold 조정 (0.6 → 0.7)
# 더 조용한 환경에서 등록
```

### 성능 최적화

#### GPU 메모리 최적화
```python
# 주기적 메모리 정리
torch.cuda.empty_cache()
```

#### 오디오 버퍼 조정
```javascript
// 프론트엔드에서 청크 크기 조정
this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
// 2048 (낮은 지연) 또는 8192 (안정성)
```

## 📊 API 명세서

### WebSocket 메시지 포맷

#### 클라이언트 → 서버

**화자 등록 시작**
```json
{
    "type": "start_speaker_registration",
    "speaker_id": 0  // 0 또는 1
}
```

**화자 등록 완료**
```json
{
    "type": "complete_speaker_registration",
    "speaker_id": 0
}
```

**실시간 전사 시작**
```json
{
    "type": "start_transcription"
}
```

**오디오 데이터**
```
Binary PCM16 데이터 (16kHz, 1채널)
```

#### 서버 → 클라이언트

**모델 로드 완료**
```json
{
    "type": "model_loaded",
    "message": "STT AI 모델 로드 완료! 화자 등록을 시작하세요."
}
```

**화자 등록 시작됨**
```json
{
    "type": "speaker_registration_started",
    "speaker_id": 0,
    "message": "화자 1번의 목소리를 등록해주세요. (3-5초간 말씀해주세요)"
}
```

**화자 등록 완료됨**
```json
{
    "type": "speaker_registration_completed",
    "speaker_id": 0,
    "message": "화자 1번 등록이 완료되었습니다."
}
```

**실시간 전사 결과**
```json
{
    "type": "transcription",
    "speaker_id": 0,           // 0 또는 1
    "text": "안녕하세요",       // 전사 결과
    "timestamp": "2025-08-21T05:30:00Z"
}
```

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 지원

문제가 발생하거나 질문이 있으시면 이슈를 생성해주세요.

---

**개발자**: 2020112534@linuxserver2  
**최종 업데이트**: 2025-08-21