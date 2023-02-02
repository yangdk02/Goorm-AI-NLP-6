# Goorm AI 자연어 처리 전문가 양성 과정 6기 Project
👥 양다경(팀장), 구선혜, 김유나, 김현수, 김현지, 이용주, 최원서

## 1️⃣ Project 1: Yelp 데이터셋을 이용한 영어 리뷰 감성 분류
- 2022.12.26~2022.12.30

### 🖐️ Summary
- Yelp 데이터셋에 다양한 사전학습 모델을 적용해 각 모델의 성능 비교
- 99%의 정확도로 Yelp 데이터셋 영어 리뷰 긍·부정 감성 분석

### 🌏 Development Environment
- Google Colab Pro+
- Pakages:

|Package|Version|
|---|---|
|datasets|2.8.0|
|matplotlib|3.2.2|
|numpy|1.21.6|
|pandas|1.3.5|
|transformers|4.25.1|
|torch|1.13.0|
|wandb|0.13.7|

### 💾 Datasets
- Yelp Dataset

### 📁 Directory Tree
~~~
📁 Project 2
│   📄 goorm-proj1.ipynb                                                  메인 코드
│   📄 goorm-proj1-baeline-colab.ipynb                                    베이스라인 코드
│   📄 Yelp 데이터셋을 이용한 영어 리뷰 감성 분류.pdf                        보고서
└───📁 data                                                                                            
│       💾 sentiment.train.0                                              훈련 데이터셋
│       💾 sentiment.train.1                                              훈련 데이터셋
│       💾 sentiment.dev.0                                                검증 데이터셋
│       💾 sentiment.dev.1                                                검증 데이터셋
│       💾 test_no_label.csv                                              테스트 데이터셋
└───📁 results
└───────📁 parameters                                                     모델 중간 결과 (Not uploaded)
        💾 submission_hard-voted.csv                                      테스트 데이터셋 최종 결과 (Hard voting 적용)
        💾 submission_model_albert-base_64_adamw_1e-05_0_3.bin.csv        ALBERT-base 테스트 데이터셋 최종 결과
~~~

## 2️⃣ Project 2: KoBigBird 사전학습 모델을 이용한 한국어 MRC
- 2023.01.02~2023.01.16

### 🖐️ Summary
- 주어진 KLUE-MRC 데이터셋에 대하여 KoBigBird 사전학습 모델을 미세 조정
- 모델이 주어진 질문을 이해하고 뉴스 기사에서 해당 질문에 대한 답을 찾을 수 있음

### 🌏 Development Environment
- Google Colab Pro+
- Pakages:

|Package|Version|
|---|---|
|datasets|2.8.0|
|levenshtein|0.20.9|
|matplotlib|3.2.2|
|numpy|1.21.6|
|pandas|1.3.5|
|transformers|4.25.1|
|torch|1.13.0|
|torchinfo|1.7.1|
|wandb|0.13.7|

### 💾 Datasets
- KLUE-MRC Dataset
- [AI-Hub 뉴스 기사 기계독해 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=577)
    - Training 라벨링데이터 中 추출형(서술형)
    
### 📁 Directory Tree
~~~
📁 Project 2
│   📄 goorm_proj2_korean-mrc.ipynb                                                                    메인 코드 (훈련)
│   📄 goorm_proj2_korean-mrc_validation-n-submission-with-error-analysis.ipynb                        메인 코드 (검증 및 후처리)
│   📄 korean-mrc-baseline-goorm-6.ipynb                                                               베이스라인 코드
│   📄 KoBigBird 사전학습 모델을 이용한 한국어 MRC.pdf                                                   보고서
└───📁 input                                                                                           
│       💾 baseline.csv                                                                                
│       💾 blank.csv                                                                                   
│       💾 test.json                                                                                   KLUE-MRC 테스트 데이터셋
│       💾 TL_span_extraction.json                                                                     AI-Hub 뉴스 기사 기계독해 데이터 (Not uploaded)
│       💾 train.json                                                                                  KLUE-MRC 훈련 데이터셋
└───📁 output
        💾 submission-kobigbird-bert-base_lr1e-05_ep30_max1024_317.csv                                 Model 3 테스트 데이터셋 최종 결과
        💾 new_submission-kobigbird-bert-base_lr1e-05_ep30_max1024_317.csv                             Model 3 테스트 데이터셋 최종 결과 (후처리 적용)
        📄 lr_1e_1_proj2.ipynb                                                                         Model 1 훈련 코드 및 결과
        📄 lr_1e_5_proj2.ipynb                                                                         Model 2 훈련 코드 및 결과
        📄 proj2_데이터_증강 (1).ipynb                                                                  Model 3 훈련 코드 및 결과
        💾 raw_predictions-kobigbird-bert-base_lr0.1_ep30_max1024_531-finetuned-klue.pkl               Model 1 검증 데이터셋 중간 결과
        💾 raw_predictions-kobigbird-bert-base_lr1e-05_ep30_max1024_13-finetuned-klue.pkl              Model 2 검증 데이터셋 중간 결과
        💾 raw_predictions-kobigbird-bert-base_lr1e-05_ep30_max1024_317-finetuned-klue.pkl             Model 3 검증 데이터셋 중간 결과
        💾 test_raw_predictions-kobigbird-bert-base_lr1e-05_ep30_max1024_317-finetuned-klue.pkl        Model 3 테스트 데이터셋 중간 결과
~~~

## 3️⃣ Project 3: 구어체-문어체 변환기
- 2023.01.17~2023.02.02

### 🖐️ Summary
- 입력된 문장이 구어체인지, 문어체인지 분류
    - 예) 정기 시험이 엄청 어려웠어요. → 구어체 / 정기 시험은 매우 어려웠습니다. → 문어체
- 구어체→문어체 & 문어체→구어체 스타일 변환
    - 예) 정기 시험이 엄청 어려웠어요. → 정기 시험은 매우 어려웠습니다. / 정기 시험은 매우 어려웠습니다. → 정기 시험이 엄청 어려웠어요.

### 🌏 Development Environment
- Google Colab Pro+
- Pakages:

|Package|Version|
|---|---|
|datasets|2.8.0|
|numpy|1.21.6|
|pandas|1.3.5|
|transformers|4.25.1|
|torch|1.13.0|
|torchinfo|1.7.1|
|wandb|0.13.7|

### 💾 Datasets
- [AI-Hub 한국어-영어 번역(병렬) 말뭉치](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=126)
    
### 📁 Directory Tree
~~~
📁 Project 3
│   📄 구어체-문어체 변환기.pdf
└───📁 codes
└───────📁 building dataset
│           📄 한국어_영어_데이터셋_나누기.ipynb                          AI-Hub 한국어-영어 번역(병렬) 말뭉치 데이터셋 분할
|           📄 Googletrans_API_실험.ipynb                               Google API를 이용한 구어체-문어체 병렬 데이터셋 구축
|           📄 한영모델제작_0125.ipynb                                   한국어 구어체→영어 번역기 (ELECTRA-KoGPT2)
|           📄 proj3_dataset_en_ko_공유.ipynb                           영어→한국어 문어체 번역기 (ELECTRA-KoGPT2)
|           📄 goorm-proj3_filtering.ipynb                              구어체-문어체 병렬 코퍼스 필터링
└───────📁 classifier
│           📄 문어_구어_분류기.ipynb                                    구어체-문어체 분류기
└───────📁 converter
│           📄 goorm-proj3_spoken-written-converter_kobart.ipynb        구어체→문어체 변환기 (KoBART)
│           📄 goorm-proj3_spoken-written-converter_kot5.ipynb          구어체→문어체 변환기 (T5)
│           📄 goorm-proj3_written-spoken-converter_kobart.ipynb        문어체→구어체 변환기 (KoBART)
│           📄 goorm-proj3_written-spoken-converter_kot5.ipynb          문어체→문어체 변환기 (T5)
│           📄 goorm-proj3_generation-v2.ipynb                          텍스트 생성
|           📄 goorm-proj3_validation.ipynb                             검증
└───────📁 demo
│           📄 spoken-written-converter.py
|           📄 requirements.txt
~~~
