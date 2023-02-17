# swyg_RecSys

> 유저의 답변을 기반으로 유저에게 맞는 취미를 추천해주는 다중 분류 기반 추천 시스템

> 활용 분야: MBTI 같은 특정 유형 테스트

## 데이터셋 구성

1. 취미별 크롤링 진행(블로그 글 약 14000개)
2. 전처리 후 문장 총 143590개 확보
3. 각 문장과 각 키워드 간의 코사인 유사도 계산

## 데이터 전처리

1. Accumulative bias를 활용한 위치 정보 임베딩
2. Vectorize 후에도 "각 범주의 점수는 { }점이다" 라는 정보가 남음


### example

```Python
score_bias = [20, 20, 20, 20]
```


```Python
score = [11, 13, 19, 10]
```

유저의 ```i+1``` 번째 범주 점수는  ```score[i]``` 점이다.

```Python
score_with_bias = []
for idx, num in enumerate(score):
  if idx==0: score_with_bias.append(0)
  else:
    score_with_bias.append(sum(score_bias[:idx]))
```

위 코드를 통해 bias와 결합된 점수를 구하면 

```score_with_bias = [11, 23, 39, 30]```
가 된다.

```Python
def vectorize_sequences(sequences, dimension=40):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        sequence = list(sequence)
        results[i, sequence] = 1.
    return results
```

마지막으로, ```vectorize_squences(score_with_bias)```를 사용하여 데이터를 벡터화하면, </br>
리스트의 각 값에 해당하는 인덱스에만 1 값이 할당되고, 나머지는 0인 길이 80의 벡터가 만들어진다.


또한 training label은, label 간의 연속성이 없기 때문에 ```one hot encoding```을 통해 학습과 추론에 용이하도록 한다.


## 학습

```Bash
python3 train.py --data_path {DATA_PATH} --epoch {EPOCH} --batch_size {BATCH_SIZE}
```


```Python
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(40,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

모델은 두 개의 은닉층을 가진 딥러닝 모델을 사용하였고, 병목을 예방하기 위해 서서히 차원을 줄여 나가서 분류 클래스만큼 표현하도록 하였다.</br>
optimizer는 국룰 ```adam```을 사용하였다.

> epoch:20, batch_size=512

> 워낙 데이터가 적어서 배치 사이즈가 저리 클 필욘 없었지만 추후 데이터 추가를 염두에 두고 512로 설정하였다.

![image](https://user-images.githubusercontent.com/86578246/213868694-7652cdb7-42e7-40d0-b89c-7a0322a2e08c.png)

역시 학습도 매우 빠르게 진행 되고, 클래스가 적다보니 학습이 안정적으로 진행이 되는걸 확인할 수 있었다.

## 추론

![image](https://user-images.githubusercontent.com/86578246/213868780-438e80ff-bf1c-44d4-9e7f-de23faba4369.png)

훈련 데이터 값을 조금씩만 바꾸고 예측을 해봤는데, 원하던 결과는 맨 첫 컬럼이 1,2,3,4,5로 나오는 것이었지만 5,2,3,4,5 로 나온 것을 보니 정확도가 나쁘지 않은 정도임을 알 수 있었다.

### Usage

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

## 진행 상황

### 1/21 데이터셋 구축 시에 사용할 KoBERT 감성 분석 모델 학습 완료
### 1/22 .py 리팩토링 및 모듈화 완료
### _1/24 base_info.json을 이용한 추가 모듈화 작업 완료_

## 추후 추가 내용
- ~~.py 파일로 바꿔서 업로드~~ 1/22
- 키워드 크롤링 파이프라인 구축
- 데이터셋 구축 및 유효성 확보/검증
- 모델 평가 및 검증(k-fold cross validation)
- 모델 경량화(Pruning, layer 축소)

