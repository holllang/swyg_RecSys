# swyg_RecSys

> 유저의 답변을 기반으로 유저에게 맞는 취미를 추천해주는 다중 분류 기반 추천 시스템

> 활용 분야: MBTI 같은 특정 유형 테스트


## 🎯 추천 로직 모식도
<img width="281" alt="홀랑 모델 로직" src="https://github.com/holllang/swyg_RecSys/assets/86578246/6a3226bd-165a-4666-9451-ebf7ff98d30d">


## 🛠 사용기술 및 라이브러리

- Tensorflow, Keras
- Python

## 🗄 데이터셋

- 취미 32가지를 선정한 후, 각 취미에 따른 검색결과를 크롤링하였다.
- 데이터를 다루기 쉽도록 크롤링한 문장들은 json형식으로 각 취미별로 정리하였다.
- 사용자의 유형은 mbti를 기반으로 추론하였으며, 이에 따라 크롤링한 문장들을 mbti와 관련한 `키워드`들로 분류하기로 하였다.
- 각 취미가 어떤 mbti와 연관성이 있는지 파악하기 위해 키워드를 선정하였고, 각 문장들과 키워드 간의 `유사도`를 판단하였다.
    - 키워드
        
        ```python
        keywords = {
                    'E': ["바깥 외향 활발"],
                    'I': ["실내 조용 혼자"],
                    'N': ["생각 상상 "],
                    'S': ["기분 느낌"],
                    'F': ["감성 공감 감정"],
                    'T': ["이성 이해"],
                    'J': ["계획 오래"],
                    'P': ["즉흥 잠깐"]
                    }
        ```
        
    - 유사도 판단(tf-idf, cosine 유사도)
        
        ```python
        def get_score(sentences, hobby, category):
            sentences = data_pre[hobby]
            s_len = len(sentences)
            compare = keywords[category]
            sentences = compare + sentences
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
            val = cosine_similarity(tfidf_matrix[0:1],tfidf_matrix[1:]).tolist()[0]
            val.sort(reverse=True)
            total = s_len - val.count(0.0)
            try:
                return sum(val)/total
            except:
                return 0
        ```
        
- mbti는 4개의 범주(E/I, N/S, F/T, J/P)로 이루어져있기 때문에, 각 범주별로 유사도를 비교하였고, `[E, N, F, J]`를 기준으로 그 유사도를 비교하여 데이터셋을 구성하였다.
    - 데이터셋 예시
         
        ```python
        {
            ...
            "캠핑": [1, 2, 1, 1],
            "서핑": [2, 1, 0, 1],
            "배드민턴": [2, 1, 1, 1],
            "유튜버 시작하기": [2, 1, 1, 1],
            "풍경사진 찍기": [2, 1, 1, 1],
            "전시회 보러가기": [2, 1, 1, 1],
            ...
        }

        ```
        

## 📈 학습

- keras의 `다중 분류 모델`을 파이프라인으로 사용하였다.
- 모델 내에서는 input이 `벡터 시퀀스`화 된 후, output을 도출한다.
- input의 범위를 bias로 설정하여 추론을 진행하기 때문에 `positional information`이 벡터 시퀀스에 남는다.

```Bash
python3 train.py --data_path {DATA_PATH} --epoch {EPOCH} --batch_size {BATCH_SIZE}
```

- 예시

    input : `[2, 1, 0, 3]`

    bias : `[4, 4, 4, 4]`

    ⇒ input with bias : `[2, 5, 8, 15]`
        
- Hyperparemeters
    - epoch : 100
    - batch size : 5
- 데이터셋의 라벨은, 취미간의 연속성이 없기 때문에 `one hot vector`를 사용하여 labeling을 진행하였다.

![Figure_1](https://user-images.githubusercontent.com/86578246/221891920-e45e58c9-9bee-45c1-99ab-a0098badffd8.png)



## 🔨 추론

```Python
from infer import InferModule
from keras import models

model = models.load_model('./model_saved')
IM = InferModule(model)

if __name__=='__main__':

    # 사용자의 문항별 답변 항목을 추론 input으로
    result = IM.start_inferring([1,2,4,1,2,3,2,2,1,2,3,1,1,2,3,2])
    print(result)
    
    # result : ['취미2', '취미4', '취미1']
```

추론 모듈을 먼저 로드해두고,</br>
추론이 필요할 땐 ```IM.start_inferring``` 으로 불러서 실행하면 된다.</br>

매 추론 요청마다 모델을 로드하는 것이 아닌,</br>
모델을 로드해두고 추론을 하는 것이기 때문에 실행 시간이 짧다 :)

### 추론 테스트 결과

1st test

input : `[2, 1, 1, 1]` ⇒ ESTP

output : `['풍경사진찍기', '전시회구경', '유튜버되기']`

2nd test

input : `[1, 2, 3, 1]` ⇒ INFP

output : `['목공예', '카톡이모티콘만들기', '헬스']`

3rd test

input : `[0, 1, 1, 1]` ⇒ ISTP

output : `['연극보기', '바이닐수집', '카톡이모티콘만들기']`

## 📌 평가

- 데이터셋은 `점수를 기준으로 정렬`되어있다.
- 만약 선택지와 유사한 취미들을 추천해준다면, 데이터셋 내에서 취미들의 `인덱스가 인접`해있을 것이라 가정하였다.
- 추론으로 나온 취미들로 `(최대 인덱스-최소 인덱스)`를 계산하여 `인접도`를 확인한다.
- 인접도의 범위는 2~31인데, 2점을 100점으로, 31점을 0점으로 정규화하여 확인한다.
- 모든 답변의 경우의 수 `4096`개를 사용하여 답변 추론을 진행한다.
- 점수화 결과

![output](https://user-images.githubusercontent.com/86578246/221891958-7df94414-0ca8-4d93-a341-8ed49511e808.png)

max: 100.0
min: 20.689655172413794
avg: 68.34506330819025

