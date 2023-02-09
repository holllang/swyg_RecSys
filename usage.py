from infer import InferModule
from keras import models
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model = models.load_model('./model_saved')
IM = InferModule(model)

if __name__=='__main__':

    # 사용자의 문항별 답변 항목을 추론 input으로
    result = IM.start_inferring([0, 53, 42, 51])
    print(result)
