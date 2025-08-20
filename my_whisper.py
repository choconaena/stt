# /home/2020112534/stt/my_whisper_improved.py

import numpy as np
import torch
import time
import webrtcvad
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import librosa
import math
from pyannote.audio import Model
from sklearn.metrics.pairwise import cosine_similarity


def compute_rms_db(audio_np):
    rms = np.sqrt(np.mean(audio_np ** 2))
    if rms < 1e-10:
        return -float("inf")
    return 20 * math.log10(rms)


def is_human_voice(audio_np, sr, n_mfcc=13, var_threshold=20.0):
    mfccs = librosa.feature.mfcc(y=audio_np, sr=sr, n_mfcc=n_mfcc)
    variances = np.var(mfccs, axis=1)
    mean_var = np.mean(variances)
    return (mean_var > var_threshold, mean_var)


class WhisperSTT:
    def __init__(self, model_path="/home/2020112534/safe_hi/model/my_whisper", device="cuda",
                 sample_rate=16000, partial_interval=1.0, min_seg_duration=1.0, silence_duration=0.5,
                 max_segment_duration=7.0, rms_threshold=-50.0, var_threshold=20.0, vad_mode=2,
                 num_speakers=2, hf_token="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"):

        self.sample_rate = sample_rate
        self.device = device
        self.min_seg_duration = min_seg_duration
        self.silence_duration = silence_duration
        self.max_segment_duration = max_segment_duration
        self.rms_threshold = rms_threshold
        self.var_threshold = var_threshold
        self.vad_mode = vad_mode

        self.num_speakers = num_speakers
        self.enable_diarization = num_speakers > 1

        # Whisper 모델 로드
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path).to(device)
        self.processor = AutoProcessor.from_pretrained(model_path)

        # VAD 설정
        self.vad = webrtcvad.Vad(mode=vad_mode)
        self.frame_duration_ms = 30
        self.vad_frame_bytes = int(self.sample_rate * 2 * self.frame_duration_ms / 1000)

        # 오디오 버퍼 및 상태 관리
        self.audio_buffer = bytearray()
        self.sentence_count = 1
        self.last_final_text = ""
        self.segment_start_time = time.time()
        self.last_log_state = None

        # 화자분리 관련 초기화
        if self.enable_diarization:
            try:
                self.embedding_model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token)
                self.embedding_model.eval()
                self.speaker_embeddings = []  # [(embedding vector, speaker_id)]
                self.initial_enroll_count = 0
                self.similarity_threshold = 0.6  # 화자 인식 임계값
                self.speaker_history = []  # 최근 화자 기록
                self.speaker_smooth_window = 3  # 화자 스무딩 윈도우
                print("화자분리 모델 로드 완료")
            except Exception as e:
                print(f"화자분리 모델 로드 실패: {e}")
                self.enable_diarization = False

    def extract_embedding(self, audio_np):
        """오디오에서 화자 임베딩 추출"""
        try:
            # 오디오 길이가 너무 짧으면 패딩
            if len(audio_np) < self.sample_rate * 0.5:
                padding_length = int(self.sample_rate * 0.5) - len(audio_np)
                audio_np = np.pad(audio_np, (0, padding_length), 'constant')
            
            # 오디오 길이가 너무 길면 자르기
            elif len(audio_np) > self.sample_rate * 10:
                audio_np = audio_np[:int(self.sample_rate * 10)]
            
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
            
            with torch.no_grad():
                embedding = self.embedding_model(audio_tensor)
            
            return embedding.squeeze().cpu().numpy()
            
        except Exception as e:
            print(f"임베딩 추출 오류: {e}")
            return None

    def assign_speaker(self, emb):
        """임베딩을 기반으로 화자 할당"""
        if emb is None:
            return 0  # 기본 화자
        
        # 초기 등록 단계
        if self.initial_enroll_count < self.num_speakers:
            speaker_id = self.initial_enroll_count
            self.speaker_embeddings.append((emb, speaker_id))
            self.initial_enroll_count += 1
            print(f"화자 {speaker_id} 등록됨 (총 {self.initial_enroll_count}/{self.num_speakers})")
            return speaker_id

        # 등록된 화자들과 유사도 계산
        if len(self.speaker_embeddings) == 0:
            return 0
        
        try:
            similarities = []
            for speaker_emb, speaker_id in self.speaker_embeddings:
                sim = cosine_similarity([emb], [speaker_emb])[0][0]
                similarities.append((sim, speaker_id))
            
            # 가장 높은 유사도를 가진 화자 선택
            best_sim, best_speaker = max(similarities, key=lambda x: x[0])
            
            # 임계값보다 낮으면 새로운 화자로 처리 (하지만 최대 화자 수 제한)
            if best_sim < self.similarity_threshold and len(self.speaker_embeddings) < self.num_speakers:
                new_speaker_id = len(self.speaker_embeddings)
                self.speaker_embeddings.append((emb, new_speaker_id))
                print(f"새로운 화자 {new_speaker_id} 감지됨 (유사도: {best_sim:.3f})")
                return new_speaker_id
            
            # 화자 스무딩 적용
            self.speaker_history.append(best_speaker)
            if len(self.speaker_history) > self.speaker_smooth_window:
                self.speaker_history.pop(0)
            
            # 최빈값으로 화자 결정
            from collections import Counter
            speaker_counts = Counter(self.speaker_history)
            smoothed_speaker = speaker_counts.most_common(1)[0][0]
            
            return smoothed_speaker
            
        except Exception as e:
            print(f"화자 할당 오류: {e}")
            return 0

    def register_speaker_embedding(self, audio_np, speaker_id):
        """특정 화자의 임베딩을 등록"""
        emb = self.extract_embedding(audio_np)
        if emb is not None:
            # 기존 등록된 화자가 있으면 업데이트
            for i, (existing_emb, existing_id) in enumerate(self.speaker_embeddings):
                if existing_id == speaker_id:
                    # 기존 임베딩과 새 임베딩의 평균 계산
                    averaged_emb = (existing_emb + emb) / 2
                    self.speaker_embeddings[i] = (averaged_emb, speaker_id)
                    print(f"화자 {speaker_id} 임베딩 업데이트됨")
                    return
            
            # 새로운 화자 등록
            self.speaker_embeddings.append((emb, speaker_id))
            if self.initial_enroll_count <= speaker_id:
                self.initial_enroll_count = speaker_id + 1
            print(f"화자 {speaker_id} 새로 등록됨")

    def _whisper_transcribe_np(self, audio_np):
        """Whisper 모델을 사용한 음성 인식"""
        try:
            inputs = self.processor(audio_np, sampling_rate=self.sample_rate, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs)
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return text.strip()
            
        except Exception as e:
            print(f"음성 인식 오류: {e}")
            return ""

    def process_chunk(self, audio_chunk):
        """오디오 청크 처리"""
        events = []
        self.audio_buffer.extend(audio_chunk)
        now = time.time()

        # VAD를 통한 음성 활동 감지
        num_frames = len(self.audio_buffer) // self.vad_frame_bytes
        silence_frames = 0
        
        for i in range(num_frames):
            frame = self.audio_buffer[i * self.vad_frame_bytes:(i + 1) * self.vad_frame_bytes]
            try:
                if not self.vad.is_speech(frame, self.sample_rate):
                    silence_frames += 1
                else:
                    silence_frames = 0
            except:
                # VAD 오류 시 음성으로 가정
                silence_frames = 0

        silence_time = silence_frames * (self.frame_duration_ms / 1000.0)

        # 세그먼트 종료 조건 확인
        should_process = (
            silence_time >= self.silence_duration or 
            (now - self.segment_start_time >= self.max_segment_duration)
        )

        if should_process:
            duration_sec = len(self.audio_buffer) / (self.sample_rate * 2)

            if duration_sec < self.min_seg_duration:
                # 너무 짧은 세그먼트는 무음 처리
                if self.enable_diarization:
                    events.append({"type": "final", "text": f"[화자 0]: [무음]"})
                else:
                    events.append({"type": "final", "text": f"[{self.sentence_count}번문장]: [무음]"})
            else:
                # 오디오 데이터 처리
                audio_np = np.frombuffer(self.audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                current_db = compute_rms_db(audio_np)
                is_voice, _ = is_human_voice(audio_np, self.sample_rate, var_threshold=self.var_threshold)

                # 필터링할 텍스트들
                skip_texts = {"응", "뭐", "뭐야.", "감사합니다.", "응.", "아", "흠", "르.", "너.", ".", "음", "어", "그", "네", "예"}

                if current_db >= self.rms_threshold and is_voice:
                    # 음성 인식 수행
                    final_text = self._whisper_transcribe_np(audio_np)
                    
                    if final_text.strip() and final_text.strip() not in skip_texts:
                        if self.enable_diarization:
                            # 화자분리 수행
                            emb = self.extract_embedding(audio_np)
                            speaker_id = self.assign_speaker(emb)
                            events.append({
                                "type": "final", 
                                "text": f"[화자 {speaker_id}]: {final_text.strip()}",
                                "speaker_id": speaker_id,
                                "confidence": current_db
                            })
                        else:
                            events.append({
                                "type": "final", 
                                "text": f"[{self.sentence_count}번문장]: {final_text.strip()}"
                            })
                    else:
                        # 의미없는 텍스트는 무음 처리
                        if self.enable_diarization:
                            events.append({"type": "final", "text": f"[화자 0]: [무음]"})
                        else:
                            events.append({"type": "final", "text": f"[{self.sentence_count}번문장]: [무음]"})
                else:
                    # 음성이 아니거나 볼륨이 낮음
                    if self.enable_diarization:
                        events.append({"type": "final", "text": f"[화자 0]: [무음]"})
                    else:
                        events.append({"type": "final", "text": f"[{self.sentence_count}번문장]: [무음]"})

            # 버퍼 초기화 및 카운터 증가
            self.audio_buffer = bytearray()
            self.sentence_count += 1
            self.segment_start_time = time.time()

        return events

    def reset_session(self):
        """세션 초기화"""
        self.audio_buffer = bytearray()
        self.sentence_count = 1
        self.last_final_text = ""
        self.segment_start_time = time.time()
        
        if self.enable_diarization:
            self.speaker_embeddings = []
            self.initial_enroll_count = 0
            self.speaker_history = []
        
        print("STT 세션이 초기화되었습니다.")

    def get_speaker_info(self):
        """등록된 화자 정보 반환"""
        if not self.enable_diarization:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "registered_speakers": len(self.speaker_embeddings),
            "max_speakers": self.num_speakers,
            "speaker_ids": [speaker_id for _, speaker_id in self.speaker_embeddings]
        }