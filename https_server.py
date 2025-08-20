#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTTPS 서버 - STT 웹페이지 서빙용
포트 8443에서 HTTPS로 HTML 파일을 서빙합니다.
"""

import http.server
import socketserver
import ssl
import os

PORT = 8444
DIRECTORY = "."

# SSL 인증서 경로 - 로컬 key 디렉토리 사용
CERT_FILE = "./key/fullchain.pem"
KEY_FILE = "./key/privkey.pem"

class MyHTTPSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def end_headers(self):
        # CORS 및 보안 헤더 추가
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        # HTTPS 전용 헤더
        self.send_header('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        super().end_headers()

def main():
    print(f"현재 디렉토리: {os.getcwd()}")
    print(f"디렉토리 내 파일들:")
    for file in os.listdir("."):
        if file.endswith(('.html', '.htm')):
            print(f"  📄 {file}")
    
    print(f"\nHTTPS 서버 시작: https://localhost:{PORT}")
    print(f"디렉토리 목록: https://localhost:{PORT}/")
    print(f"테스트 페이지: https://localhost:{PORT}/test.html")
    if os.path.exists("speaker_stt_frontend.html"):
        print(f"STT 페이지: https://localhost:{PORT}/speaker_stt_frontend.html")
    print(f"외부 접속: https://safe-hi.xyz:{PORT}/")
    print("Ctrl+C로 종료")
    
    # SSL 컨텍스트 설정
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    try:
        context.load_cert_chain(certfile=CERT_FILE, keyfile=KEY_FILE)
    except FileNotFoundError:
        print("❌ SSL 인증서를 찾을 수 없습니다.")
        print("다음 중 하나를 시도해보세요:")
        print("1. 로컬 인증서 경로로 변경:")
        print("   CERT_FILE = './key/fullchain.pem'")
        print("   KEY_FILE = './key/privkey.pem'")
        print("2. HTTP 서버 사용: python -m http.server 8080")
        return
    
    with socketserver.TCPServer(("", PORT), MyHTTPSRequestHandler) as httpd:
        # SSL 래핑
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n서버 종료됨")
        except Exception as e:
            print(f"❌ 서버 오류: {e}")

if __name__ == "__main__":
    main()