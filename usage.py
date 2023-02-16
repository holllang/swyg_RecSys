from infer import InferModule
from keras import models
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model = models.load_model('./model_saved')
IM = InferModule(model)

if __name__=='__main__':

    # 사용자의 문항별 답변 항목을 추론 input으로
    result = IM.start_inferring([2, 1, 1, 1])
    print(result)


# 질문들의 답변을 기반으로 각 mbti 범주별 퍼센테이지를 계산
# 각 범주별 퍼센테이지가 [30, 60, 100, 30] 으로 결과가 나왔다면
# 0~19 범위로 정규화 시킨다 -> [5, 11, 19, 5]
# 이렇게 정규화시킨 값이 모델의 입력값이 됨