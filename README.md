# swyg_RecSys

* [주피터 노트북](https://github.com/swyg-goorm/swyg_RecSys/blob/main/SWYG_RecSys.ipynb) 보러가기

> 유저의 답변을 기반으로 유저에게 맞는 취미를 추천해주는 다중 분류 기반 추천 시스템

> 활용 분야: MBTI 같은 특정 유형 테스트

## 데이터셋 구성

> 1. 앞으로 만들어질 취미 별 키워드를 기반으로 크롤링 진행</br>
> 2. 감성 평가 모델을 사용한 취미 별 키워드 점수화</br>
> 3. EDA 결과를 활용하여 문항과 답변에 맞는 데이터셋 구축</br>

## 데이터 전처리

> Accumulative bias를 활용한 위치 정보 임베딩</br>
> Vectorize 후에도 "몇번 문항에 몇번 답변을 골랐다" 라는 정보가 남음</br>


### example

```
num_per_question = [2,3,4,3,2,3,3,2,3,2,3,2,2,2,2,2]
```

```i+1``` 번째 문항의 답변 개수는 총 ```num_per_question[i]``` 개이다.

```
answer = [1,2,4,1,2,3,3,2,1,2,3,1,1,2,3,2]
```

유저는 ```i+1``` 번째 문항에  ```answer[i]``` 번째 답변을 골랐다.

```
question_bias = []
for idx, num in enumerate(num_per_question):
  if idx==0: question_bias.append(0)
  else:
    question_bias.append(sum(num_per_question[:idx]))
```

위 코드를 통해 bias를 구하면 

```question_bias = [0,2,5,9,12,14,17,20,22,25,27,30,32,34,36,38]```

answer와 question_bias를 같은 인덱스끼리 더하고 1을 빼면(0~39 인덱싱),

```answer_with_bias = [0,3,8,9,13,16,19,21,22,26,29,30,32,35,38,39]```

```
def vectorize_sequences(sequences, dimension=40):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        sequence = list(sequence)
        results[i, sequence] = 1.
    return results
```

마지막으로, ```vectorize_squences(answer_with_bias)```를 사용하여 데이터를 벡터화하면, </br>
리스트의 각 값에 해당하는 인덱스에만 1 값이 할당되고, 나머지는 0인 길이 40의 벡터가 만들어진다.


또한 training label은, label 간의 연속성이 없기 때문에 ```one hot encoding```을 통해 학습과 추론에 용이하도록 한다.


## 학습

```
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(40,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

모델은 두 개의 은닉층을 가진 딥러닝 모델을 사용하였고, 병목을 예방하기 위해 서서히 차원을 줄여 나가서 분류 클래스만큼 표현하도록 하였다.
optimizer는 국룰 adam을 사용하였다.

> epoch:20, batch_size=512

> 워낙 데이터가 적어서 배치 사이즈가 저리 클 필욘 없었지만 추후 데이터 추가를 염두에 두고 512로 설정하였다.

![image](https://user-images.githubusercontent.com/86578246/213868694-7652cdb7-42e7-40d0-b89c-7a0322a2e08c.png)

역시 학습도 매우 빠르게 진행 되고, 클래스가 적다보니 학습이 안정적으로 진행이 되는걸 확인할 수 있었다.

## 추론

![image](https://user-images.githubusercontent.com/86578246/213868780-438e80ff-bf1c-44d4-9e7f-de23faba4369.png)

훈련 데이터 값을 조금씩만 바꾸고 예측을 해봤는데, 원하던 결과는 맨 첫 컬럼이 1,2,3,4,5로 나오는 것이었지만 5,2,3,4,5 로 나온 것을 보니 정확도가 나쁘지 않은 정도임을 알 수 있었다.

## 진행 상황

### _1/21 데이터셋 구축 시에 사용할 KoBERT 감성 분석 모델 학습 완료_

## 추후 추가 내용
- .py 파일로 바꿔서 업로드
- 데이터셋 구축 및 유효성 검증



