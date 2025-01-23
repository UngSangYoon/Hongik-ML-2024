# 저조도 CCTV 영상에서의 객체 탐지 및 상황 요약 🚦🔍
본 프로젝트는 컴퓨터 비전(CV)과 자연어 처리(NLP)를 결합하여 저조도 CCTV 영상에서 실시간 객체 탐지 및 상황 요약을 가능하게 합니다.
YOLO 모델을 활용한 객체 탐지와 Fine-Tuned LLM을 통해 야간 감시 작업에 효율적이고 해석 가능한 결과를 제공합니다.

## [📹 프로젝트 발표 영상](https://youtu.be/zRkAB4p7g0k?si=ApTYE96mgMpkHh0h)

## 🎯 프로젝트 배경
저조도 CCTV 영상은 특정 순간의 사건을 실시간으로 파악하기 어려운 문제를 가지고 있습니다. 이를 해결하기 위해 다음 두 가지 목표를 설정하였습니다:
1. **저조도 환경에서의 객체 탐지**: YOLO 모델을 활용하여 저조도 Video에서 객체를 탐지하고 좌표를 추적.
2. **영상 요약**: 추적 데이터를 기반으로 영상 장면을 요약할 수 있는 소형 언어 모델(sLLM)을 개발.

## ✨ 주요 기능
- **YOLOv8 기반 객체 탐지**: 저조도 CCTV 영상에서 실시간 객체 추적.
- **영상-텍스트 처리**: 매 10초 간격으로 장면 요약 생성.
- **효율적인 학습 파이프라인**:
  - LoRA(저랭크 어댑터)를 활용한 비용 효율적인 언어 모델 미세 조정.
  - Gemini Flash를 활용한 맞춤형 학습 데이터 생성.

## 🏗 아키텍처
1. **YOLOv8 객체 탐지**:
   - YOLOv8n(나노) 모델을 저조도 데이터셋으로 학습.
   - 데이터셋: [AI-Hub 저조도 데이터셋 (2,036,575 이미지, 51 객체)](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71377)
   - 실시간 성능에 최적화.
2. **영상-텍스트 처리**:
   - 추적된 객체 데이터에는 ID, 바운딩 박스, 프레임 간격 정보 포함.
   - Microsoft Phi-2 (2.7B 파라미터) 모델을 LoRA로 미세 조정하여 요약 생성.
3. **모델 통합**:
   - 멀티프로세싱을 활용하여 YOLO와 sLLM 병렬 실행.
   - 대표 프레임에서 요약 데이터를 생성하고 간략한 출력 파일에 저장.
  
## 🤖 모델
- [Trained YOLOv8n](https://huggingface.co/Hongik-ML-2024/yolov8n-all_data-10_epochs-AdamW)
  - 최고 mAP를 기록한 모델: AdamW 옵티마이저를 사용한 YOLOv8n
  - 성능
    - 0.2 seconds (5 frame per second) 단위로 이미지 객체 탐지.
    - 객체 ID를 통해 동일 객체 추적.

- [Fine-Tuned sLLM](https://huggingface.co/Hongik-ML-2024/cctv-llm)
    - Base 모델: Microsoft Phi-2 (2.7B 파라미터), LoRA로 미세 조정.
    - 성능:
      - 약 10초 간격으로 요약 생성.
      - 간결하고 정보가 명확한 문장 출력.

## ⚠️ 한계점
- 모델 성능: YOLO는 저조도 환경에서 객체 탐지 정확도가 한정적.
- 추론 속도: 실시간 요약은 GPU 성능에 제한을 받음.
- 데이터 문제: 생성된 학습 데이터로 인해 텍스트 요약에서 간혹 상상된 정보가 포함됨.
- 저조한 정확도: 영상 내 환경을 담지 못함. 추적된 객체의 좌표만으로 상황을 요약하기 때문에 정확한 상황 설명에 한계

## 🔮 향후 방향
- 저조도 데이터셋 개선을 통한 탐지 정확도 향상.
- 상황 요약에 필요한 더 많은 정보를 처리.
- 언어 모델의 추론 속도 최적화.
- 완전 자동화된 CCTV 감시 시스템으로 확장 가능성 탐구.
  
## 🤝 기여자
- 윤웅상 (Yolo Training data preprocessing, Video to Text Processing, CV-Language Model integration)
- 정진홍 (LLM Traing data preprocessing, Training Data Generation(By Gemini Flash model), LLM Training)
- 박채연 (Yolo training Hyperparameter Tuning)
- 오현수 (Yolo training Hyperparameter Tuning)
- 유세은 (Yolo training Hyperparameter Tuning)
