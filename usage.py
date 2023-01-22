from infer import InferModule
from keras import models

model = models.load_model('./model_saved')
num_per_question = [2,3,4,3,2,3,3,2,3,2,3,2,2,2,2,2]
num2hobby = {0: '취미1', 1: '취미2', 2: '취미3', 3: '취미4', 4: '취미5'}

IM = InferModule(model, num_per_question, num2hobby)

if __name__=='__main__':

    result = IM.start_inferring([1,2,4,1,2,3,2,2,1,2,3,1,1,2,3,2])
    print(result)
