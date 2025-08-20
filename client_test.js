// 클라이언트 테스트 스크립트 (Node.js 환경용)
const WebSocket = require('ws');
const fs = require('fs');

class STTTestClient {
    constructor() {
        this.ws = null;
        this.isConnected = false;
    }

    connect() {
        // WSS 연결 (실제 서버 주소로 변경)
        this.ws = new WebSocket('wss://safe-hi.xyz:8085', {
            rejectUnauthorized: false // 개발용 (운영에서는 제거)
        });

        this.ws.on('open', () => {
            console.log('✅ 서버에 연결되었습니다.');
            this.isConnected = true;
            
            // 메타데이터 전송
            this.sendMetadata();
        });

        this.ws.on('message', (data) => {
            try {
                const message = JSON.parse(data.toString());
                this.handleMessage(message);
            } catch (e) {
                console.log('📝 텍스트 메시지:', data.toString());
            }
        });

        this.ws.on('close', () => {
            console.log('❌ 서버 연결이 끊어졌습니다.');
            this.isConnected = false;
        });

        this.ws.on('error', (error) => {
            console.error('🚨 연결 오류:', error.message);
        });
    }

    sendMetadata() {
        const metadata = {
            reportid: `test_${Date.now()}`,
            email: 'test@example.com'
        };
        
        this.ws.send(JSON.stringify(metadata));
        console.log('📤 메타데이터 전송:', metadata);
    }

    handleMessage(message) {
        console.log('📨 서버 메시지:', message);
        
        switch (message.type) {
            case 'model_loaded':
                console.log('🤖 모델 로드 완료');
                this.showMenu();
                break;
                
            case 'speaker_registration_started':
                console.log(`🎤 화자 ${message.speaker_id + 1} 등록 시작`);
                break;
                
            case 'speaker_registration_completed':
                console.log(`✅ 화자 ${message.speaker_id + 1} 등록 완료`);
                this.showMenu();
                break;
                
            case 'transcription_started':
                console.log('🚀 실시간 전사 시작');
                break;
                
            case 'transcription':
                console.log(`💬 [화자 ${message.speaker_id + 1}]: ${message.text}`);
                break;
        }
    }

    showMenu() {
        console.log('\n📋 사용 가능한 명령:');
        console.log('1. register0 - 화자 1 등록');
        console.log('2. register1 - 화자 2 등록');
        console.log('3. start - 실시간 전사 시작');
        console.log('4. quit - 종료');
        console.log('명령을 입력하세요:');
    }

    registerSpeaker(speakerId) {
        if (!this.isConnected) {
            console.log('❌ 서버에 연결되지 않았습니다.');
            return;
        }

        this.ws.send(JSON.stringify({
            type: 'start_speaker_registration',
            speaker_id: speakerId
        }));

        // 5초 후 등록 완료
        setTimeout(() => {
            this.ws.send(JSON.stringify({
                type: 'complete_speaker_registration',
                speaker_id: speakerId
            }));
        }, 5000);
    }

    startTranscription() {
        if (!this.isConnected) {
            console.log('❌ 서버에 연결되지 않았습니다.');
            return;
        }

        this.ws.send(JSON.stringify({
            type: 'start_transcription'
        }));
    }

    sendTestAudio() {
        // 테스트용 더미 오디오 데이터 전송
        const dummyAudio = Buffer.alloc(1024, 0);
        if (this.isConnected) {
            this.ws.send(dummyAudio);
        }
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
        }
    }
}

// 실행
const client = new STTTestClient();
client.connect();

// 키보드 입력 처리
const readline = require('readline');
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

rl.on('line', (input) => {
    const command = input.trim().toLowerCase();
    
    switch (command) {
        case 'register0':
            console.log('🎤 화자 1 등록 중... (5초)');
            client.registerSpeaker(0);
            break;
            
        case 'register1':
            console.log('🎤 화자 2 등록 중... (5초)');
            client.registerSpeaker(1);
            break;
            
        case 'start':
            console.log('🚀 실시간 전사 시작');
            client.startTranscription();
            // 테스트용 오디오 전송 시작
            setInterval(() => {
                client.sendTestAudio();
            }, 100);
            break;
            
        case 'quit':
            console.log('👋 연결을 종료합니다.');
            client.disconnect();
            rl.close();
            process.exit(0);
            break;
            
        default:
            console.log('❓ 알 수 없는 명령입니다.');
            client.showMenu();
    }
});

// 프로세스 종료 시 정리
process.on('SIGINT', () => {
    console.log('\n👋 프로그램을 종료합니다.');
    client.disconnect();
    rl.close();
    process.exit(0);
});