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


class StreamingTranscriber:
    """실시간 스트리밍 전사 클래스"""
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        self.partial_buffer = []
        self.final_buffer = []
        self.last_partial_text = ""
        self.last_final_text = ""
        
    def transcribe_streaming(self, audio_np, sample_rate, is_final=False):
        """스트리밍 방식으로 전사"""
        try:
            # 버퍼에 추가
            if is_final:
                # 최종일 때는 전체 버퍼 사용
                self.final_buffer.extend(self.partial_buffer)
                self.final_buffer.append(audio_np)
                full_audio = np.concatenate(self.final_buffer) if self.final_buffer else audio_np
                
                # 전체 오디오로 최종 전사
                text = self._transcribe(full_audio, sample_rate)
                
                # 버퍼 초기화
                self.partial_buffer = []
                self.final_buffer = []
                self.last_final_text = text
                
                return text, True
            else:
                # 부분 전사
                self.partial_buffer.append(audio_np)
                
                # 최근 3초만 사용하여 부분 전사
                if len(self.partial_buffer) > 3:
                    self.partial_buffer = self.partial_buffer[-3:]
                
                partial_audio = np.concatenate(self.partial_buffer)
                text = self._transcribe(partial_audio, sample_rate)
                
                # 이전과 다른 경우만 반환
                if text != self.last_partial_text:
                    self.last_partial_text = text
                    return text, False
                
                return None, False
                
        except Exception as e:
            print(f"스트리밍 전사 오류: {e}")
            return None, False
    
    def _transcribe(self, audio_np, sample_rate):
        """실제 전사 수행"""
        # 최소 길이 확인
        min_samples = int(sample_rate * 30)
        if len(audio_np) < min_samples:
            padded_audio = np.zeros(min_samples, dtype=np.float32)
            padded_audio[:len(audio_np)] = audio_np
            audio_np = padded_audio
        
        # 정규화
        audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-8)
        
        inputs = self.processor(
            audio_np,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=2,  # 빠른 처리를 위해 줄임
                temperature=0.7,
                do_sample=True
            )
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return text.strip()


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


class WhisperSTT:
    def __init__(self, model_path="/home/2020112534/safe_hi/model/my_whisper", device="cuda",
                 sample_rate=16000, min_seg_duration=0.8, silence_duration=0.4,
                 max_segment_duration=5.0, rms_threshold=-50.0, var_threshold=15.0, vad_mode=2,
                 num_speakers=2, hf_token="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"):

        self.sample_rate = sample_rate
        self.device = device
        self.min_seg_duration = min_seg_duration  # 0.8초로 완화
        self.silence_duration = silence_duration  # 0.4초로 완화
        self.max_segment_duration = max_segment_duration
        self.rms_threshold = rms_threshold  # 원래 값으로 복원
        self.var_threshold = var_threshold  # 더 낮은 값으로 완화
        self.vad_mode = vad_mode  # 중간 수준으로 복원

        self.num_speakers = num_speakers
        self.enable_diarization = num_speakers > 1

        # Whisper 모델 로드
        print("Whisper 모델 로딩 중...")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path).to(device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        print("Whisper 모델 로드 완료")

        # VAD 설정
        self.vad = webrtcvad.Vad(mode=vad_mode)
        self.frame_duration_ms = 20  # 더 짧은 프레임으로 세밀한 검출
        self.vad_frame_bytes = int(self.sample_rate * 2 * self.frame_duration_ms / 1000)

        # 개선된 버퍼 관리
        self.audio_buffer = bytearray()
        self.processing_queue = Queue(maxsize=10)
        self.result_queue = Queue()
        
        # 슬라이딩 윈도우 버퍼 (중첩 처리용)
        self.sliding_buffer = deque(maxlen=int(sample_rate * 2))  # 2초 버퍼
        self.overlap_ratio = 0.3  # 30% 중첩
        
        # 화자분리 관련 초기화
        if self.enable_diarization:
            try:
                print("화자분리 모델 로딩 중...")
                self.embedding_model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token)
                self.embedding_model.eval()
                self.speaker_embeddings = []
                self.similarity_threshold = 0.55  # 더 낮은 임계값으로 민감도 증가
                
                # 청크 기반 화자 세그먼터
                self.chunk_segmenter = None  # 화자 등록 후 초기화
                
                # 화자별 활성 상태 추적
                self.speaker_active = [False] * num_speakers
                self.last_speaker_time = [0] * num_speakers
                
                print("화자분리 모델 로드 완료")
            except Exception as e:
                print(f"화자분리 모델 로드 실패: {e}")
                self.enable_diarization = False

        # 스트리밍 전사기 초기화
        self.streaming_transcriber = StreamingTranscriber(self.model, self.processor, self.device)
        
        # 무음 추적기
        self.silence_tracker = SilenceTracker(threshold_seconds=1.5)
        
        # 처리 스레드 시작
        self.is_running = True  # is_running을 스레드 시작 전에 설정
        self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.processing_thread.start()
        
        # 상태 관리
        self.last_processing_time = time.time()

    def extract_embedding(self, audio_np):
        """오디오에서 화자 임베딩 추출 (개선된 버전)"""
        try:
            # 오디오 정규화
            audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-8)
            
            # 길이 조정
            min_length = int(self.sample_rate * 0.3)  # 최소 0.3초
            max_length = int(self.sample_rate * 5)    # 최대 5초
            
            if len(audio_np) < min_length:
                audio_np = np.pad(audio_np, (0, min_length - len(audio_np)), 'constant')
            elif len(audio_np) > max_length:
                audio_np = audio_np[:max_length]
            
            # 3D 텐서로 변환 (경고 해결)
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
            
            with torch.no_grad():
                embedding = self.embedding_model(audio_tensor)
            
            return embedding.squeeze().cpu().numpy()
            
        except Exception as e:
            print(f"임베딩 추출 오류: {e}")
            return None

    def assign_speaker_with_overlap(self, audio_np, timestamp):
        """중첩 상황을 고려한 화자 할당"""
        if not self.enable_diarization:
            return 0
        
        emb = self.extract_embedding(audio_np)
        if emb is None:
            return 0
        
        # 모든 등록된 화자와 유사도 계산
        similarities = []
        for speaker_emb, speaker_id in self.speaker_embeddings:
            sim = cosine_similarity([emb], [speaker_emb])[0][0]
            similarities.append((speaker_id, sim))
        
        if not similarities:
            return 0
        
        # 유사도가 임계값을 넘는 모든 화자 찾기
        active_speakers = [(sid, sim) for sid, sim in similarities if sim > self.similarity_threshold]
        
        if not active_speakers:
            # 가장 유사한 화자 선택
            best_speaker = max(similarities, key=lambda x: x[1])
            return best_speaker[0]
        
        # 여러 화자가 활성화된 경우, 최근 활동과 유사도를 고려
        now = time.time()
        best_score = -1
        best_speaker = 0
        
        for speaker_id, similarity in active_speakers:
            # 최근 활동 가중치 계산
            time_weight = 1.0 / (1.0 + (now - self.last_speaker_time[speaker_id]))
            
            # 종합 점수 = 유사도 * 시간 가중치
            score = similarity * (0.7 + 0.3 * time_weight)
            
            if score > best_score:
                best_score = score
                best_speaker = speaker_id
        
        # 화자 활동 시간 업데이트
        self.last_speaker_time[best_speaker] = now
        self.speaker_active[best_speaker] = True
        
        return best_speaker

    def register_speaker_embedding(self, audio_np, speaker_id):
        """특정 화자의 임베딩을 등록 (개선된 버전)"""
        # 여러 세그먼트에서 임베딩 추출
        segment_length = int(self.sample_rate * 1.0)  # 1초 세그먼트
        embeddings = []
        
        for i in range(0, len(audio_np) - segment_length, segment_length // 2):
            segment = audio_np[i:i + segment_length]
            emb = self.extract_embedding(segment)
            if emb is not None:
                embeddings.append(emb)
        
        if embeddings:
            # 평균 임베딩 계산
            avg_embedding = np.mean(embeddings, axis=0)
            
            # 기존 화자 업데이트 또는 새로 추가
            updated = False
            for i, (existing_emb, existing_id) in enumerate(self.speaker_embeddings):
                if existing_id == speaker_id:
                    # 기존 임베딩과 새 임베딩의 가중 평균
                    self.speaker_embeddings[i] = (
                        0.7 * existing_emb + 0.3 * avg_embedding,
                        speaker_id
                    )
                    updated = True
                    break
            
            if not updated:
                self.speaker_embeddings.append((avg_embedding, speaker_id))
            
            print(f"화자 {speaker_id} 임베딩 등록/업데이트 완료 (샘플 수: {len(embeddings)})")
            
            # 청크 세그먼터 초기화
            if len(self.speaker_embeddings) >= 2:
                self.chunk_segmenter = ChunkBasedSpeakerSegmenter(
                    self.embedding_model,
                    self.speaker_embeddings,
                    self.sample_rate
                )

    def _whisper_transcribe_np(self, audio_np):
        """Whisper 모델을 사용한 음성 인식 (개선된 버전)"""
        try:
            # 최소 길이 확인 (Whisper는 최소 30초 오디오 필요)
            min_samples = int(self.sample_rate * 30)  # 30초
            
            # 오디오가 너무 짧으면 패딩
            if len(audio_np) < min_samples:
                # 무음으로 패딩
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
                    max_new_tokens=128,
                    num_beams=3,
                    temperature=0.7,
                    do_sample=True
                )
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return text.strip()
            
        except Exception as e:
            print(f"음성 인식 오류: {e}")
            return ""

    def _processing_worker(self):
        """백그라운드 처리 스레드 - 청크 기반 화자분리"""
        chunk_id = 0
        
        while self.is_running:
            try:
                segment = self.processing_queue.get(timeout=0.1)
                if segment is None:
                    continue
                
                # 오디오 길이 확인
                audio_duration = len(segment.audio_data) / self.sample_rate
                
                # 너무 짧은 세그먼트는 건너뛰기
                if audio_duration < 0.5:
                    print(f"세그먼트가 너무 짧음: {audio_duration:.2f}초")
                    continue
                
                # 청크 기반 화자 분리가 활성화된 경우
                if self.enable_diarization and self.chunk_segmenter:
                    # 오디오를 청크로 분할 (0.5초 단위)
                    chunk_size = int(self.sample_rate * 0.5)
                    chunks = []
                    
                    for i in range(0, len(segment.audio_data), chunk_size):
                        chunk = segment.audio_data[i:i + chunk_size]
                        if len(chunk) >= chunk_size * 0.5:  # 최소 0.25초
                            chunks.append(chunk)
                    
                    # 각 청크 처리
                    for chunk in chunks:
                        speaker, confidence = self.chunk_segmenter.process_chunk(chunk, chunk_id)
                        chunk_id += 1
                    
                    # 화자별 세그먼트 생성
                    speaker_segments = self.chunk_segmenter.create_speaker_segments()
                    
                    # 각 세그먼트에 대해 음성 인식
                    for seg in speaker_segments:
                        text = self._whisper_transcribe_np(seg['audio'])
                        
                        if text and len(text) > 1:
                            result = {
                                "type": "final",
                                "text": text,
                                "speaker_id": seg['speaker'],
                                "timestamp": segment.start_time,
                                "confidence": seg['confidence'],
                                "num_chunks": seg['num_chunks']
                            }
                            
                            self.result_queue.put(result)
                            print(f"[화자 {seg['speaker']}] {text} (청크: {seg['num_chunks']}개)")
                    
                    # 청크 버퍼 초기화
                    self.chunk_segmenter.reset()
                    
                else:
                    # 기존 방식 (화자분리 없음)
                    text = self._whisper_transcribe_np(segment.audio_data)
                    
                    if text and len(text) > 1:
                        result = {
                            "type": "final",
                            "text": text,
                            "speaker_id": 0,
                            "timestamp": segment.start_time,
                            "confidence": segment.confidence
                        }
                        
                        self.result_queue.put(result)
                    
            except Empty:
                continue
            except Exception as e:
                print(f"처리 스레드 오류: {e}")
                import traceback
                traceback.print_exc()

    def process_chunk(self, audio_chunk):
        """오디오 청크 처리 (개선된 무음 처리)"""
        events = []
        self.audio_buffer.extend(audio_chunk)
        
        # 슬라이딩 버퍼 업데이트
        audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        self.sliding_buffer.extend(audio_np)
        
        now = time.time()
        
        # 스트리밍 전사 (0.3초마다)
        if now - self.last_processing_time >= 0.3 and len(self.sliding_buffer) > self.sample_rate * 0.5:
            buffer_array = np.array(self.sliding_buffer)
            
            # 부분 전사 시도
            partial_text, is_final = self.streaming_transcriber.transcribe_streaming(
                buffer_array, self.sample_rate, is_final=False
            )
            
            if partial_text:
                events.append({
                    "type": "partial",
                    "text": partial_text,
                    "timestamp": now
                })
            
            self.last_processing_time = now
        
        # VAD 기반 세그먼트 검출
        num_frames = len(self.audio_buffer) // self.vad_frame_bytes
        
        if num_frames > 0:
            speech_frames = 0
            silence_frames = 0
            
            for i in range(num_frames):
                frame = self.audio_buffer[i * self.vad_frame_bytes:(i + 1) * self.vad_frame_bytes]
                try:
                    if self.vad.is_speech(frame, self.sample_rate):
                        speech_frames += 1
                        silence_frames = 0
                    else:
                        silence_frames += 1
                except:
                    pass
            
            # 음성 활동 비율 계산
            speech_ratio = speech_frames / max(num_frames, 1)
            silence_time = silence_frames * (self.frame_duration_ms / 1000.0)
            
            # 무음 추적 업데이트
            
            is_long_silence, silence_duration = self.silence_tracker.update(
                speech_ratio < 0.1, now
            )
            

            # 세그먼트 종료 조건
            buffer_duration = len(self.audio_buffer) / (self.sample_rate * 2)
            
            should_process = False
            process_reason = ""
            
            if buffer_duration >= self.max_segment_duration:
                should_process = True
                process_reason = "max_duration"
            elif silence_time >= self.silence_duration and buffer_duration >= self.min_seg_duration:
                should_process = True
                process_reason = "silence_detected"
            elif is_long_silence and buffer_duration >= 0.5:
                # 긴 무음은 조용한 음성일 수 있으므로 처리
                should_process = True
                process_reason = "long_silence_might_be_speech"
            
            if should_process:
                if buffer_duration >= 0.5:  # 최소 0.5초
                    # 세그먼트 생성
                    audio_data = np.frombuffer(self.audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # 파형 패턴 분석
                    is_voice, confidence = analyze_waveform_pattern(audio_data, self.sample_rate)
                    
                    # 긴 무음이면 임계값 낮춤
                    confidence_threshold = 0.3 if process_reason == "long_silence_might_be_speech" else 0.4
                    
                    if is_voice and confidence > confidence_threshold:
                        segment = AudioSegment(
                            audio_data=audio_data,
                            start_time=now - buffer_duration,
                            end_time=now
                        )
                        segment.confidence = confidence
                        
                        try:
                            self.processing_queue.put_nowait(segment)
                        except:
                            pass
                    elif confidence > 0:
                        print(f"노이즈로 판단됨 (신뢰도: {confidence:.2f}, 이유: {process_reason})")
                
                # 버퍼 초기화
                self.audio_buffer = bytearray()
        
        # 결과 수집
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
                events.append(result)
            except Empty:
                break
        
        return events

    def _process_sliding_window(self):
        """슬라이딩 윈도우 처리 (중첩 감지용)"""
        if len(self.sliding_buffer) < self.sample_rate * 0.5:  # 최소 0.5초
            return
        
        # 슬라이딩 윈도우를 여러 작은 세그먼트로 분할
        window_size = int(self.sample_rate * 0.5)  # 0.5초 윈도우
        step_size = int(window_size * (1 - self.overlap_ratio))  # 중첩 고려
        
        audio_array = np.array(self.sliding_buffer)
        
        for i in range(0, len(audio_array) - window_size, step_size):
            segment = audio_array[i:i + window_size]
            
            # RMS 체크
            if compute_rms_db(segment) < self.rms_threshold:
                continue
            
            # 음성 특징 체크
            is_voice, _ = is_human_voice(segment, self.sample_rate, self.var_threshold)
            if not is_voice:
                continue
            
            # 화자 감지 (처리 큐를 거치지 않고 직접)
            if self.enable_diarization:
                emb = self.extract_embedding(segment)
                if emb is not None:
                    # 각 화자와의 유사도 계산
                    for speaker_emb, speaker_id in self.speaker_embeddings:
                        sim = cosine_similarity([emb], [speaker_emb])[0][0]
                        
                        # 화자별 추적기 업데이트
                        if speaker_id < len(self.speaker_trackers):
                            self.speaker_trackers[speaker_id].update(speaker_id, sim)

    def reset_session(self):
        """세션 초기화"""
        self.audio_buffer = bytearray()
        self.sliding_buffer.clear()
        
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
            self.speaker_active = [False] * self.num_speakers
            self.last_speaker_time = [0] * self.num_speakers
            self.chunk_segmenter = None  # 세그먼터 초기화
        
        print("STT 세션이 초기화되었습니다.")

    def cleanup(self):
        """리소스 정리"""
        self.is_running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)

    def get_speaker_info(self):
        """등록된 화자 정보 반환"""
        if not self.enable_diarization:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "registered_speakers": len(self.speaker_embeddings),
            "max_speakers": self.num_speakers,
            "speaker_ids": [speaker_id for _, speaker_id in self.speaker_embeddings],
            "active_speakers": [i for i, active in enumerate(self.speaker_active) if active]
        }