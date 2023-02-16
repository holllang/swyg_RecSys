import numpy as np
import json

def vectorize_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        sequence = list(sequence)
        results[i, sequence] = 1.
    return results

class InferModule:
    def __init__(self, model):
        self.model = model
        self.score_bias = []
        with open('./model_saved/base_info.json', encoding='utf8') as f:
            base_info = json.load(f)
            self.position_score = base_info["list"]
            self.num2hobby = base_info["hobby_enum"]

        for idx in range(len(self.position_score)):
            if idx == 0: 
                self.score_bias.append(0)
            else:
                self.score_bias.append(sum(self.position_score[:idx]))

    def start_inferring(self, infer_score):
        score_add = []
        for s in infer_score:
            if s < 10: score_add.append(0)
            else: score_add.append(1)
        infer_score = score_add + infer_score
        X_infer = [(a+b-1) for a, b in zip(infer_score, self.score_bias)]
        X_infer = vectorize_sequences([X_infer], sum(self.position_score))
        predictions = self.model.predict(X_infer)
        
        for pred in predictions:
            ind = np.argpartition(pred, -3)[-3:]
            ind = ind[np.argsort(pred[ind])][::-1]
            hobby = []
            for i in ind:
                hobby.append(self.num2hobby[str(i)])
            return hobby