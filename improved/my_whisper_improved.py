import numpy as np
import torch
import time
import webrtcvad
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import librosa
import math
from pyannote.audio import Model
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque
import threading
from queue import Queue, Empty
import traceback

'''임시로 추가함 -chat gpt'''
# 호출부와 시그니처/리턴을 맞춘 SilenceTracker
class SilenceTracker:
    def __init__(self, threshold_seconds=1.5):
        self.threshold_seconds = threshold_seconds
        self.last_speech_time = time.time()

    def update(self, is_speech: bool, now: float):
        """is_speech=True면 최근 발화 시점 갱신. (is_long_silence, duration) 반환"""
        if is_speech:
            self.last_speech_time = now
        duration = now - self.last_speech_time
        return (duration > self.threshold_seconds, duration)

from dataclasses import dataclass
@dataclass
class AudioSegment:
    audio_data: np.ndarray
    start_time: float
    end_time: float
    confidence: float = 0.0   # 기본값, 나중에 코드에서 덮어씀
''''''

def analyze_waveform_pattern(audio_np, sample_rate=16000):
    """파형 패턴 분석으로 실제 음성인지 판별"""
    if len(audio_np) < sample_rate * 0.1:  # 최소 0.1초
        return False, 0.0
    
    # 1. 영점 교차율 (Zero Crossing Rate) 계산
    zero_crossings = np.where(np.diff(np.sign(audio_np)))[0]
    zcr = len(zero_crossings) / len(audio_np)
    
    # 2. 단기 에너지 변화율 계산
    frame_length = int(sample_rate * 0.025)  # 25ms 프레임
    hop_length = int(sample_rate * 0.010)    # 10ms 홉
    
    energies = []
    for i in range(0, len(audio_np) - frame_length, hop_length):
        frame = audio_np[i:i + frame_length]
        energy = np.sum(frame ** 2)
        energies.append(energy)
    
    if len(energies) > 1:
        energy_var = np.var(energies)
        energy_std = np.std(energies)
        energy_mean = np.mean(energies)
        cv = energy_std / (energy_mean + 1e-10)
    else:
        cv = 0
    
    # 3. 스펙트럼 중심 계산
    fft = np.fft.rfft(audio_np)
    magnitude = np.abs(fft)
    freqs = np.fft.rfftfreq(len(audio_np), 1/sample_rate)
    
    spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)
    
    # 4. 음성 주파수 대역 에너지 비율
    voice_band_mask = (freqs >= 300) & (freqs <= 3400)
    voice_energy = np.sum(magnitude[voice_band_mask] ** 2)
    total_energy = np.sum(magnitude ** 2)
    voice_ratio = voice_energy / (total_energy + 1e-10)
    
    # 5. 시간 도메인 자기상관
    autocorr = np.correlate(audio_np, audio_np, mode='same')
    autocorr = autocorr[len(autocorr)//2:]
    
    min_lag = int(sample_rate / 400)
    max_lag = int(sample_rate / 50)
    
    if max_lag < len(autocorr):
        pitch_autocorr = autocorr[min_lag:max_lag]
        if len(pitch_autocorr) > 0:
            max_autocorr = np.max(pitch_autocorr) / (autocorr[0] + 1e-10)
        else:
            max_autocorr = 0
    else:
        max_autocorr = 0
    
    # 6. 판별 점수 계산
    score = 0
    
    if 0.01 < zcr < 0.1:
        score += 1
    if cv > 0.5:
        score += 2
    if 500 < spectral_centroid < 2500:
        score += 1
    if voice_ratio > 0.6:
        score += 2
    if max_autocorr > 0.3:
        score += 2
    
    is_speech = score >= 4
    confidence = score / 8.0
    
    return is_speech, confidence


def compute_rms_db(audio_np):
    rms = np.sqrt(np.mean(audio_np ** 2))
    if rms < 1e-10:
        return -float("inf")
    return 20 * math.log10(rms)


def is_human_voice(audio_np, sr, var_threshold=20.0):
    """개선된 음성 판별 - 파형 패턴 분석 추가"""
    try:
        mfccs = librosa.feature.mfcc(y=audio_np, sr=sr, n_mfcc=13)
        variances = np.var(mfccs, axis=1)
        mean_var = np.mean(variances)
        mfcc_is_voice = mean_var > var_threshold
        
        pattern_is_voice, pattern_confidence = analyze_waveform_pattern(audio_np, sr)
        
        is_voice = mfcc_is_voice or (pattern_is_voice and pattern_confidence > 0.5)
        confidence = max(mean_var / 100.0, pattern_confidence)
        
        return (is_voice, confidence)
    except:
        return (False, 0.0)


class ChunkBasedSpeakerSegmenter:
    """청크 기반 화자 세그먼테이션 클래스"""
    def __init__(self, embedding_model, speaker_embeddings, sample_rate=16000):
        self.embedding_model = embedding_model
        self.speaker_embeddings = speaker_embeddings
        self.sample_rate = sample_rate
        self.chunk_buffer = []
        self.chunk_speakers = []
        self.chunk_confidences = []
        
        # 파라미터
        self.min_chunk_duration = 0.5  # 최소 청크 길이 (초)
        self.merge_threshold = 0.3  # 병합 임계값 (낮은 신뢰도)
        self.split_threshold = 0.7  # 분리 임계값 (높은 신뢰도)
        self.min_segment_chunks = 3  # 최소 세그먼트 청크 수
        
    def process_chunk(self, audio_chunk, chunk_id):
        """단일 청크 처리 및 화자 할당"""
        # 청크에서 임베딩 추출
        emb = self._extract_chunk_embedding(audio_chunk)
        if emb is None:
            return None, 0
        
        # 모든 화자와 유사도 계산
        speaker_scores = {}
        for speaker_emb, speaker_id in self.speaker_embeddings:
            similarity = cosine_similarity([emb], [speaker_emb])[0][0]
            speaker_scores[speaker_id] = similarity
        
        if not speaker_scores:
            return None, 0
        
        # 가장 높은 점수의 화자
        best_speaker = max(speaker_scores.items(), key=lambda x: x[1])
        
        # 청크 정보 저장
        self.chunk_buffer.append({
            'id': chunk_id,
            'audio': audio_chunk,
            'speaker': best_speaker[0],
            'confidence': best_speaker[1],
            'all_scores': speaker_scores
        })
        
        return best_speaker[0], best_speaker[1]
    
    def _extract_chunk_embedding(self, audio_chunk):
        """청크에서 임베딩 추출"""
        try:
            # 정규화
            audio_np = audio_chunk / (np.max(np.abs(audio_chunk)) + 1e-8)
            
            # 최소 길이 확인
            min_length = int(self.sample_rate * self.min_chunk_duration)
            if len(audio_np) < min_length:
                audio_np = np.pad(audio_np, (0, min_length - len(audio_np)), 'constant')
            
            # 3D 텐서로 변환
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                embedding = self.embedding_model(audio_tensor)
            
            return embedding.squeeze().cpu().numpy()
            
        except Exception as e:
            print(f"청크 임베딩 추출 오류: {e}")
            return None
    
    def create_speaker_segments(self):
        """청크 버퍼를 기반으로 화자 세그먼트 생성"""
        if len(self.chunk_buffer) < self.min_segment_chunks:
            return []
        
        segments = []
        current_segment = {
            'chunks': [self.chunk_buffer[0]],
            'speaker': self.chunk_buffer[0]['speaker'],
            'start_chunk': 0,
            'avg_confidence': self.chunk_buffer[0]['confidence']
        }
        
        for i in range(1, len(self.chunk_buffer)):
            chunk = self.chunk_buffer[i]
            prev_chunk = self.chunk_buffer[i-1]
            
            # 연속성 판단 로직
            should_merge = self._should_merge_chunks(
                current_segment, 
                chunk, 
                prev_chunk
            )
            
            if should_merge:
                # 현재 세그먼트에 청크 추가
                current_segment['chunks'].append(chunk)
                # 평균 신뢰도 업데이트
                total_conf = sum(c['confidence'] for c in current_segment['chunks'])
                current_segment['avg_confidence'] = total_conf / len(current_segment['chunks'])
                
                # 화자가 다르면 다수결로 화자 재결정
                if chunk['speaker'] != current_segment['speaker']:
                    speaker_votes = {}
                    for c in current_segment['chunks']:
                        speaker_votes[c['speaker']] = speaker_votes.get(c['speaker'], 0) + c['confidence']
                    
                    # 가중 투표로 최종 화자 결정
                    best_speaker = max(speaker_votes.items(), key=lambda x: x[1])
                    current_segment['speaker'] = best_speaker[0]
            else:
                # 새로운 세그먼트 시작
                # 현재 세그먼트가 충분히 길면 저장
                if len(current_segment['chunks']) >= self.min_segment_chunks:
                    segments.append(self._finalize_segment(current_segment))
                elif segments and len(current_segment['chunks']) < 2:
                    # 너무 짧은 세그먼트는 이전 세그먼트에 병합
                    segments[-1]['chunks'].extend(current_segment['chunks'])
                    segments[-1] = self._finalize_segment(segments[-1])
                else:
                    segments.append(self._finalize_segment(current_segment))
                
                # 새 세그먼트 시작
                current_segment = {
                    'chunks': [chunk],
                    'speaker': chunk['speaker'],
                    'start_chunk': i,
                    'avg_confidence': chunk['confidence']
                }
        
        # 마지막 세그먼트 처리
        if current_segment['chunks']:
            if len(current_segment['chunks']) >= self.min_segment_chunks:
                segments.append(self._finalize_segment(current_segment))
            elif segments:
                # 마지막 세그먼트가 짧으면 이전과 병합
                segments[-1]['chunks'].extend(current_segment['chunks'])
                segments[-1] = self._finalize_segment(segments[-1])
        
        return segments
    
    def _should_merge_chunks(self, segment, current_chunk, prev_chunk):
        """청크를 현재 세그먼트에 병합할지 결정"""
        
        # 1. 같은 화자이고 높은 신뢰도면 병합
        if current_chunk['speaker'] == segment['speaker'] and current_chunk['confidence'] > self.split_threshold:
            return True
        
        # 2. 다른 화자지만 낮은 신뢰도면 병합 (노이즈일 가능성)
        if current_chunk['speaker'] != segment['speaker'] and current_chunk['confidence'] < self.merge_threshold:
            return True
        
        # 3. 연속적으로 다른 화자가 나타나는지 확인
        consecutive_different = 0
        for j in range(len(segment['chunks']) - 1, max(0, len(segment['chunks']) - 3), -1):
            if segment['chunks'][j]['speaker'] != segment['speaker']:
                consecutive_different += 1
        
        # 연속적으로 2개 이상 다른 화자면 분리
        if current_chunk['speaker'] != segment['speaker']:
            if consecutive_different >= 1 and current_chunk['confidence'] > 0.5:
                return False  # 분리
        
        # 4. 세그먼트가 너무 짧으면 병합 선호
        if len(segment['chunks']) < self.min_segment_chunks:
            return True
        
        # 5. 기본적으로는 신뢰도 기반 결정
        return current_chunk['confidence'] < self.split_threshold
    
    def _finalize_segment(self, segment):
        """세그먼트 최종 처리"""
        # 모든 청크의 오디오 결합
        combined_audio = np.concatenate([c['audio'] for c in segment['chunks']])
        
        # 최종 화자 확정 (가중 투표)
        speaker_votes = {}
        for chunk in segment['chunks']:
            for speaker_id, score in chunk['all_scores'].items():
                speaker_votes[speaker_id] = speaker_votes.get(speaker_id, 0) + score
        
        final_speaker = max(speaker_votes.items(), key=lambda x: x[1])[0]
        
        return {
            'audio': combined_audio,
            'speaker': final_speaker,
            'confidence': segment['avg_confidence'],
            'num_chunks': len(segment['chunks']),
            'chunk_ids': [c['id'] for c in segment['chunks']]
        }
    
    def reset(self):
        """버퍼 초기화"""
        self.chunk_buffer = []
        self.chunk_speakers = []
        self.chunk_confidences = []



# my_whisper_improved.py - 완전히 개선된 버전

class EnhancedStreamingTranscriber:
    """향상된 실시간 스트리밍 전사 클래스"""
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        
        # 세그먼트별 관리
        self.segments = {}  # segment_id: {'text': str, 'audio': np.array, 'is_final': bool}
        self.current_segment_id = 0
        self.active_audio_buffer = []
        
        # 텍스트 연속성 관리
        self.previous_texts = deque(maxlen=3)  # 이전 3개 텍스트 저장
        self.last_confirmed_text = ""
        self.pending_text = ""
        
        # 타이밍 관리
        self.last_transcription_time = time.time()
        self.segment_start_time = time.time()
        
    def process_audio_chunk(self, audio_np, is_silence=False):
        """오디오 청크 처리 - 연속적인 텍스트 생성"""
        current_time = time.time()
        
        # 활성 버퍼에 추가
        self.active_audio_buffer.append(audio_np)
        
        # 버퍼 길이 계산
        buffer_audio = np.concatenate(self.active_audio_buffer) if self.active_audio_buffer else audio_np
        buffer_duration = len(buffer_audio) / 16000  # 16kHz 가정
        
        results = []
        
        # 부분 전사 조건: 0.3초 이상이고 마지막 전사 후 0.2초 경과
        if buffer_duration >= 0.3 and (current_time - self.last_transcription_time) >= 0.2:
            partial_text = self._transcribe_partial(buffer_audio)
            
            if partial_text and partial_text != self.pending_text:
                # 텍스트 연속성 체크
                continuous_text = self._ensure_continuity(partial_text)
                
                if continuous_text:
                    self.pending_text = continuous_text
                    results.append({
                        "type": "partial",
                        "text": continuous_text,
                        "segment_id": self.current_segment_id,
                        "timestamp": current_time
                    })
                    self.last_transcription_time = current_time
        
        # 세그먼트 종료 조건
        should_finalize = False
        
        if is_silence and buffer_duration >= 0.5:
            # 무음이고 충분한 길이
            should_finalize = True
        elif buffer_duration >= 3.0:
            # 최대 길이 도달
            should_finalize = True
        elif is_silence and buffer_duration >= 0.3 and (current_time - self.segment_start_time) >= 1.0:
            # 짧은 무음이지만 충분한 시간 경과
            should_finalize = True
        
        if should_finalize and self.active_audio_buffer:
            # 최종 전사
            final_text = self._transcribe_final(buffer_audio)
            
            if final_text:
                # 이전 텍스트와 중복 제거
                final_text = self._remove_duplication(final_text)
                
                if final_text:
                    self.segments[self.current_segment_id] = {
                        'text': final_text,
                        'audio': buffer_audio,
                        'is_final': True
                    }
                    
                    results.append({
                        "type": "final",
                        "text": final_text,
                        "segment_id": self.current_segment_id,
                        "timestamp": current_time
                    })
                    
                    # 다음 세그먼트 준비
                    self.last_confirmed_text = final_text
                    self.previous_texts.append(final_text)
                    self.current_segment_id += 1
                    self.active_audio_buffer = []
                    self.pending_text = ""
                    self.segment_start_time = current_time
        
        return results
    
    def _ensure_continuity(self, new_text):
        """텍스트 연속성 보장"""
        if not self.last_confirmed_text:
            return new_text
        
        # 이전 확정 텍스트와의 관계 확인
        if new_text.startswith(self.last_confirmed_text):
            # 이전 텍스트의 연속
            return new_text[len(self.last_confirmed_text):].strip()
        elif self.last_confirmed_text in new_text:
            # 중간에 포함된 경우
            idx = new_text.index(self.last_confirmed_text)
            return new_text[idx + len(self.last_confirmed_text):].strip()
        else:
            # 완전히 새로운 텍스트
            return new_text
    
    def _remove_duplication(self, text):
        """중복 텍스트 제거"""
        if not self.previous_texts:
            return text
        
        # 최근 텍스트들과 비교
        for prev_text in self.previous_texts:
            if text == prev_text:
                return ""  # 완전 중복
            if text.startswith(prev_text):
                text = text[len(prev_text):].strip()
            if prev_text in text:
                text = text.replace(prev_text, "").strip()
        
        return text
    
    def _transcribe_partial(self, audio_np):
        """빠른 부분 전사"""
        try:
            # 패딩
            if len(audio_np) < 16000 * 0.5:
                return ""
            
            # 30초로 패딩
            target_length = 16000 * 30
            if len(audio_np) < target_length:
                padded = np.zeros(target_length, dtype=np.float32)
                padded[:len(audio_np)] = audio_np[:target_length]
                audio_np = padded
            
            # 정규화
            audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-8)
            
            inputs = self.processor(
                audio_np,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=50,  # 짧게
                    num_beams=1,  # 빠르게
                    do_sample=False,
                    temperature=0.7
                )
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return text.strip()
        except Exception as e:
            print(f"부분 전사 오류: {e}")
            return ""
    
    def _transcribe_final(self, audio_np):
        """정확한 최종 전사"""
        try:
            # 30초로 패딩
            target_length = 16000 * 30
            if len(audio_np) < target_length:
                padded = np.zeros(target_length, dtype=np.float32)
                padded[:len(audio_np)] = audio_np[:target_length]
                audio_np = padded
            
            # 정규화
            audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-8)
            
            inputs = self.processor(
                audio_np,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    num_beams=3,
                    do_sample=True,
                    temperature=0.7
                )
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return text.strip()
        except Exception as e:
            print(f"최종 전사 오류: {e}")
            return ""


class ImprovedWhisperSTT:
    """완전히 개선된 WhisperSTT 클래스"""
    def __init__(self, model_path="/home/2020112534/safe_hi/model/my_whisper", 
                 device="cuda", sample_rate=16000, **kwargs):
        
        self.sample_rate = sample_rate
        self.device = device
        
        # 모델 로드
        print("Whisper 모델 로딩 중...")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path).to(device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        print("Whisper 모델 로드 완료")
        
        # VAD 설정 (민감도 높임)
        self.vad = webrtcvad.Vad(mode=3)  # 0-3, 높을수록 민감
        self.frame_duration_ms = 20
        self.vad_frame_bytes = int(self.sample_rate * 2 * self.frame_duration_ms / 1000)
        
        # 향상된 스트리밍 전사기
        self.transcriber = EnhancedStreamingTranscriber(self.model, self.processor, self.device)
        
        # 버퍼 관리 - 더 엄격한 기준
        self.audio_buffer = bytearray()
        self.silence_frames = 0
        self.speech_frames = 0
        self.last_speech_time = time.time()
        
        # 품질 기준 강화
        self.min_chunk_duration = 1.0      # 최소 1초 (기존 0.8초에서 증가)
        self.min_meaningful_duration = 1.5  # 의미있는 최소 길이
        self.silence_duration = 0.6        # 무음 기준 (기존 0.4초에서 증가)
        self.max_segment_duration = 4.0    # 최대 길이 (기존 5초에서 감소)
        
        # RMS 임계값 강화 (더 큰 소리만 인식)
        self.rms_threshold = -45.0         # 기존 -50에서 -45로 상향
        self.var_threshold = 20.0          # 음성 특징 임계값
        
        # 화자분리 설정
        self.enable_diarization = kwargs.get('num_speakers', 2) > 1
        if self.enable_diarization:
            self._setup_diarization(kwargs.get('hf_token'))
        
        # 실시간 처리 큐
        self.processing_queue = Queue(maxsize=20)
        self.result_queue = Queue()
        
        # 처리 스레드
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.processing_thread.start()
        
        # 상태 관리
        self.last_processing_time = time.time()
        self.consecutive_short_segments = 0  # 연속된 짧은 세그먼트 카운트

    def _setup_diarization(self, hf_token):
        """화자분리 설정"""
        try:
            print("화자분리 모델 로딩 중...")
            self.embedding_model = Model.from_pretrained(
                "pyannote/embedding", 
                use_auth_token=hf_token
            )
            self.embedding_model.eval()
            self.speaker_embeddings = []
            self.current_speaker = 0
            print("화자분리 모델 로드 완료")
        except Exception as e:
            print(f"화자분리 모델 로드 실패: {e}")
            self.enable_diarization = False
    
    def process_chunk(self, audio_chunk):
        """메인 청크 처리 메서드 - 더 엄격한 필터링"""
        events = []
        current_time = time.time()
        
        # 버퍼에 추가
        self.audio_buffer.extend(audio_chunk)
        audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        
        # VAD로 음성/무음 감지
        is_speech = self._detect_speech_improved(audio_chunk)
        
        if is_speech:
            self.speech_frames += 1
            self.silence_frames = 0
            self.last_speech_time = current_time
        else:
            self.silence_frames += 1
        
        # 버퍼 분석
        buffer_duration = len(self.audio_buffer) / (self.sample_rate * 2)
        silence_duration = self.silence_frames * self.frame_duration_ms / 1000.0
        speech_ratio = self.speech_frames / max(self.speech_frames + self.silence_frames, 1)
        
        # 더 엄격한 세그먼트 종료 조건
        should_process = False
        process_reason = ""
        
        # 1. 최대 길이 도달
        if buffer_duration >= self.max_segment_duration:
            should_process = True
            process_reason = "max_duration"
        
        # 2. 충분한 길이 + 무음 감지 + 음성 비율 체크
        elif (buffer_duration >= self.min_chunk_duration and 
              silence_duration >= self.silence_duration and 
              speech_ratio > 0.3):  # 최소 30% 이상이 음성이어야 함
            should_process = True
            process_reason = "silence_with_speech"
        
        # 3. 매우 긴 무음 (하지만 최소 길이는 만족해야 함)
        elif (buffer_duration >= self.min_meaningful_duration and 
              silence_duration >= 1.0):  # 1초 이상 무음
            should_process = True
            process_reason = "long_silence"
        
        if should_process:
            # 오디오 품질 체크
            audio_data = np.frombuffer(self.audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
            
            # RMS (음량) 체크
            rms_db = compute_rms_db(audio_data)
            if rms_db < self.rms_threshold:
                print(f"음량 부족으로 무시: {rms_db:.1f}dB < {self.rms_threshold}dB")
                self._reset_buffer()
                return events
            
            # 음성 특징 체크
            is_voice, voice_confidence = is_human_voice(audio_data, self.sample_rate, self.var_threshold)
            
            # 더 엄격한 음성 판별 기준
            min_confidence = 0.4 if process_reason == "long_silence" else 0.5
            
            if not is_voice or voice_confidence < min_confidence:
                print(f"음성 아님으로 판단: confidence={voice_confidence:.3f}, reason={process_reason}")
                self._reset_buffer()
                return events
            
            # 연속된 짧은 세그먼트 방지
            if buffer_duration < self.min_meaningful_duration:
                self.consecutive_short_segments += 1
                if self.consecutive_short_segments >= 3:  # 연속 3회 이상 짧은 세그먼트면 무시
                    print(f"연속된 짧은 세그먼트로 무시: {buffer_duration:.2f}초")
                    self._reset_buffer()
                    return events
            else:
                self.consecutive_short_segments = 0  # 긴 세그먼트면 카운터 리셋
            
            # 품질 체크 통과 - 세그먼트 생성
            segment = AudioSegment(
                audio_data=audio_data,
                start_time=current_time - buffer_duration,
                end_time=current_time
            )
            segment.confidence = voice_confidence
            
            try:
                self.processing_queue.put_nowait(segment)
                print(f"세그먼트 큐에 추가: {buffer_duration:.2f}초, 신뢰도: {voice_confidence:.3f}, 이유: {process_reason}")
            except:
                print("처리 큐가 가득참")
            
            # 버퍼 초기화
            self._reset_buffer()
        
        # 버퍼가 너무 커지면 강제 정리 (8초 제한)
        elif buffer_duration > 8.0:
            print(f"버퍼 크기 초과로 정리: {buffer_duration:.2f}초")
            # 최근 4초만 보존
            keep_bytes = self.sample_rate * 2 * 4
            self.audio_buffer = bytearray(self.audio_buffer[-keep_bytes:])
        
        # 결과 수집
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
                events.append(result)
            except Empty:
                break
        
        return events
    
    def _detect_speech_improved(self, audio_chunk):
        """개선된 음성 감지"""
        try:
            # PCM16 형식으로 변환
            if len(audio_chunk) < self.vad_frame_bytes:
                return False
            
            # 여러 프레임에서 음성 비율 계산
            speech_frames = 0
            total_frames = 0
            
            for i in range(0, len(audio_chunk) - self.vad_frame_bytes, self.vad_frame_bytes):
                frame = audio_chunk[i:i + self.vad_frame_bytes]
                total_frames += 1
                
                try:
                    if self.vad.is_speech(frame, self.sample_rate):
                        speech_frames += 1
                except:
                    continue
            
            # 음성 비율이 30% 이상이면 음성으로 판단
            if total_frames > 0:
                speech_ratio = speech_frames / total_frames
                return speech_ratio >= 0.3
            
            return False
        except:
            return False
    
    def _reset_buffer(self):
        """버퍼 초기화"""
        self.audio_buffer = bytearray()
        self.silence_frames = 0
        self.speech_frames = 0
    
    def _processing_worker(self):
        """백그라운드 처리 스레드 - 품질 기준 적용"""
        while self.is_running:
            try:
                segment = self.processing_queue.get(timeout=0.1)
                if segment is None:
                    continue
                
                # 오디오 길이 재확인
                audio_duration = len(segment.audio_data) / self.sample_rate
                
                # 너무 짧은 세그먼트는 여기서도 한번 더 걸러냄
                if audio_duration < 0.8:
                    print(f"처리 단계에서 짧은 세그먼트 무시: {audio_duration:.2f}초")
                    continue
                
                # 음성 인식 수행
                text = self._whisper_transcribe_np(segment.audio_data)
                
                # 텍스트 품질 체크
                if text and len(text.strip()) > 3:  # 최소 4글자 이상
                    # 화자 할당
                    speaker_id = self._assign_speaker(segment.audio_data) if self.enable_diarization else 0
                    
                    result = {
                        "type": "final",
                        "text": text,
                        "speaker_id": speaker_id,
                        "timestamp": segment.start_time,
                        "confidence": segment.confidence,
                        "duration": audio_duration
                    }
                    
                    self.result_queue.put(result)
                    print(f"[화자 {speaker_id}] 처리 완료: {text[:50]}...")
                else:
                    print(f"텍스트 품질 부족으로 무시: '{text}'")
                    
            except Empty:
                continue
            except Exception as e:
                print(f"처리 스레드 오류: {e}")
                import traceback
                traceback.print_exc()

    def _whisper_transcribe_np(self, audio_np):
        """Whisper 모델을 사용한 음성 인식"""
        try:
            # 최소 길이 확인 (Whisper는 최소 30초 오디오 필요)
            min_samples = int(self.sample_rate * 30)  # 30초
            
            # 오디오가 너무 짧으면 패딩
            if len(audio_np) < min_samples:
                padded_audio = np.zeros(min_samples, dtype=np.float32)
                padded_audio[:len(audio_np)] = audio_np
                audio_np = padded_audio
            
            # 오디오 정규화
            audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-8)
            
            inputs = self.processor(
                audio_np, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=64,  # 짧게 제한 (기존 128에서 64로)
                    num_beams=2,        # 빠른 처리 (기존 3에서 2로)
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return text.strip()
            
        except Exception as e:
            print(f"음성 인식 오류: {e}")
            return ""

    def _assign_speaker(self, audio_np):
        """화자 할당"""
        if not self.enable_diarization or len(self.speaker_embeddings) == 0:
            return 0
        
        try:
            # 임베딩 추출
            emb = self.extract_embedding(audio_np)
            if emb is None:
                return self.current_speaker
            
            # 유사도 계산
            best_speaker = 0
            best_similarity = -1
            
            for speaker_emb, speaker_id in self.speaker_embeddings:
                sim = cosine_similarity([emb], [speaker_emb])[0][0]
                if sim > best_similarity:
                    best_similarity = sim
                    best_speaker = speaker_id
            
            # 임계값 체크 (0.5)
            if best_similarity > 0.5:
                self.current_speaker = best_speaker
                return best_speaker
            else:
                return self.current_speaker
                
        except Exception as e:
            print(f"화자 할당 오류: {e}")
            return 0

    def extract_embedding(self, audio_np):
        """화자 임베딩 추출"""
        try:
            # 정규화 및 길이 조정
            audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-8)
            
            min_length = int(self.sample_rate * 0.5)
            if len(audio_np) < min_length:
                audio_np = np.pad(audio_np, (0, min_length - len(audio_np)), 'constant')
            
            # 텐서 변환
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                embedding = self.embedding_model(audio_tensor)
            
            return embedding.squeeze().cpu().numpy()
        except Exception as e:
            print(f"임베딩 추출 오류: {e}")
            return None

    def register_speaker_embedding(self, audio_np, speaker_id):
        """화자 임베딩 등록"""
        emb = self.extract_embedding(audio_np)
        if emb is not None:
            # 기존 화자 업데이트 또는 추가
            updated = False
            for i, (existing_emb, existing_id) in enumerate(self.speaker_embeddings):
                if existing_id == speaker_id:
                    # 평균으로 업데이트
                    self.speaker_embeddings[i] = (
                        (existing_emb + emb) / 2,
                        speaker_id
                    )
                    updated = True
                    break
            
            if not updated:
                self.speaker_embeddings.append((emb, speaker_id))
            
            print(f"화자 {speaker_id} 임베딩 등록 완료")

    def reset_session(self):
        """세션 초기화"""
        self._reset_buffer()
        
        # 큐 비우기
        while not self.processing_queue.empty():
            try:
                self.processing_queue.get_nowait()
            except:
                break
        
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except:
                break
        
        if self.enable_diarization:
            self.speaker_embeddings = []
            self.current_speaker = 0
        
        self.consecutive_short_segments = 0
        print("STT 세션이 초기화되었습니다.")

    def cleanup(self):
        """리소스 정리"""
        self.is_running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        print("리소스 정리 완료")
    """완전히 개선된 WhisperSTT 클래스"""
    def __init__(self, model_path="/home/2020112534/safe_hi/model/my_whisper", 
                 device="cuda", sample_rate=16000, **kwargs):
        
        self.sample_rate = sample_rate
        self.device = device
        
        # 모델 로드
        print("Whisper 모델 로딩 중...")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path).to(device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        print("Whisper 모델 로드 완료")
        
        # VAD 설정 (민감도 높임)
        self.vad = webrtcvad.Vad(mode=2)  # 0-3, 높을수록 민감
        self.frame_duration_ms = 20
        self.vad_frame_bytes = int(self.sample_rate * 2 * self.frame_duration_ms / 1000)
        
        # 향상된 스트리밍 전사기
        self.transcriber = EnhancedStreamingTranscriber(self.model, self.processor, self.device)
        
        # 버퍼 관리
        self.audio_buffer = bytearray()
        self.silence_frames = 0
        self.speech_frames = 0
        self.last_speech_time = time.time()
        
        # 화자분리 설정
        self.enable_diarization = kwargs.get('num_speakers', 2) > 1
        if self.enable_diarization:
            self._setup_diarization(kwargs.get('hf_token'))
        
        # 실시간 처리 큐
        self.processing_queue = Queue(maxsize=20)
        self.result_queue = Queue()
        
        # 처리 스레드
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.processing_thread.start()
    
    def _setup_diarization(self, hf_token):
        """화자분리 설정"""
        try:
            print("화자분리 모델 로딩 중...")
            self.embedding_model = Model.from_pretrained(
                "pyannote/embedding", 
                use_auth_token=hf_token
            )
            self.embedding_model.eval()
            self.speaker_embeddings = []
            self.current_speaker = 0
            print("화자분리 모델 로드 완료")
        except Exception as e:
            print(f"화자분리 모델 로드 실패: {e}")
            self.enable_diarization = False
    
    def process_chunk(self, audio_chunk):
        """메인 청크 처리 메서드"""
        events = []
        current_time = time.time()
        
        # 버퍼에 추가
        self.audio_buffer.extend(audio_chunk)
        audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        
        # VAD로 음성/무음 감지
        is_speech = self._detect_speech(audio_chunk)
        
        if is_speech:
            self.speech_frames += 1
            self.silence_frames = 0
            self.last_speech_time = current_time
        else:
            self.silence_frames += 1
        
        # 버퍼 처리 조건
        buffer_duration = len(self.audio_buffer) / (self.sample_rate * 2)
        silence_duration = self.silence_frames * self.frame_duration_ms / 1000.0
        
        # 스트리밍 전사기로 처리
        if buffer_duration >= 0.2:  # 최소 0.2초
            buffer_np = np.frombuffer(self.audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
            
            # 무음 여부 전달
            is_silence = silence_duration >= 0.2
            
            # 전사 결과 얻기
            transcription_results = self.transcriber.process_audio_chunk(
                buffer_np, 
                is_silence=is_silence
            )
            
            for result in transcription_results:
                # 화자 할당
                if self.enable_diarization and len(self.speaker_embeddings) > 0:
                    result['speaker_id'] = self._assign_speaker(buffer_np)
                else:
                    result['speaker_id'] = 0
                
                events.append(result)
                
                # 최종 텍스트인 경우 버퍼 초기화
                if result['type'] == 'final':
                    self.audio_buffer = bytearray()
                    self.silence_frames = 0
                    self.speech_frames = 0
        
        # 너무 긴 버퍼 방지 (5초 제한)
        if buffer_duration > 5.0:
            self.audio_buffer = bytearray(self.audio_buffer[-(self.sample_rate * 2 * 4):])
        
        return events
    
    def _detect_speech(self, audio_chunk):
        """VAD를 사용한 음성 감지"""
        try:
            # PCM16 형식으로 변환
            if len(audio_chunk) < self.vad_frame_bytes:
                return False
            
            frame = audio_chunk[:self.vad_frame_bytes]
            return self.vad.is_speech(frame, self.sample_rate)
        except:
            return False
    
    def _assign_speaker(self, audio_np):
        """화자 할당"""
        if not self.enable_diarization or len(self.speaker_embeddings) == 0:
            return 0
        
        try:
            # 임베딩 추출
            emb = self.extract_embedding(audio_np)
            if emb is None:
                return self.current_speaker
            
            # 유사도 계산
            best_speaker = 0
            best_similarity = -1
            
            for speaker_emb, speaker_id in self.speaker_embeddings:
                sim = cosine_similarity([emb], [speaker_emb])[0][0]
                if sim > best_similarity:
                    best_similarity = sim
                    best_speaker = speaker_id
            
            # 임계값 체크 (0.5)
            if best_similarity > 0.5:
                self.current_speaker = best_speaker
                return best_speaker
            else:
                return self.current_speaker
                
        except Exception as e:
            print(f"화자 할당 오류: {e}")
            return 0
    
    def extract_embedding(self, audio_np):
        """화자 임베딩 추출"""
        try:
            # 정규화 및 길이 조정
            audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-8)
            
            min_length = int(self.sample_rate * 0.5)
            if len(audio_np) < min_length:
                audio_np = np.pad(audio_np, (0, min_length - len(audio_np)), 'constant')
            
            # 텐서 변환
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                embedding = self.embedding_model(audio_tensor)
            
            return embedding.squeeze().cpu().numpy()
        except Exception as e:
            print(f"임베딩 추출 오류: {e}")
            return None
    
    def register_speaker_embedding(self, audio_np, speaker_id):
        """화자 임베딩 등록"""
        emb = self.extract_embedding(audio_np)
        if emb is not None:
            # 기존 화자 업데이트 또는 추가
            updated = False
            for i, (existing_emb, existing_id) in enumerate(self.speaker_embeddings):
                if existing_id == speaker_id:
                    # 평균으로 업데이트
                    self.speaker_embeddings[i] = (
                        (existing_emb + emb) / 2,
                        speaker_id
                    )
                    updated = True
                    break
            
            if not updated:
                self.speaker_embeddings.append((emb, speaker_id))
            
            print(f"화자 {speaker_id} 임베딩 등록 완료")
    
    def _processing_worker(self):
        """백그라운드 처리 워커"""
        while self.is_running:
            try:
                # 큐에서 작업 가져오기
                task = self.processing_queue.get(timeout=0.1)
                if task:
                    # 작업 처리
                    pass
            except Empty:
                continue
            except Exception as e:
                print(f"처리 워커 오류: {e}")
    
    def reset_session(self):
        """세션 초기화"""
        self.audio_buffer = bytearray()
        self.transcriber = EnhancedStreamingTranscriber(self.model, self.processor, self.device)
        self.silence_frames = 0
        self.speech_frames = 0
        self.current_speaker = 0
        if self.enable_diarization:
            self.speaker_embeddings = []
        print("세션 초기화 완료")
    
    def cleanup(self):
        """리소스 정리"""
        self.is_running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        print("리소스 정리 완료")