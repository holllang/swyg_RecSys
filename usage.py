from infer import InferModule
from keras import models

# 추론 모듈 세팅을 위해 필수적으로 필요한 파라미터들
model = models.load_model('./model_saved')
num_per_question = [2,3,4,3,2,3,3,2,3,2,3,2,2,2,2,2]
num2hobby = {0: '취미1', 1: '취미2', 2: '취미3', 3: '취미4', 4: '취미5'}

IM = InferModule(model, num_per_question, num2hobby)

if __name__=='__main__':

    result = IM.start_inferring([1,2,4,1,2,3,2,2,1,2,3,1,1,2,3,2])
    print(result)
