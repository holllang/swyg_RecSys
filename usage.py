from infer import InferModule
from keras import models
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model = models.load_model('./model_saved.h5')
IM = InferModule(model)

if __name__=='__main__':

    # 사용자의 문항별 답변 항목을 추론 input으로
    inferringResponse = IM.start_inferring([1, 2, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1])
    print(inferringResponse)
    



# 질문들의 답변을 기반으로 각 mbti 범주별 퍼센테이지를 계산
# 범주별 퍼센테이지를 0~3 범위로 정규화 시킨 후 모델에 input으로 줌