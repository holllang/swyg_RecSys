import json

class DataLoader:
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf8') as f:
            self.datadict_from_json =  json.load(f)
        
        self.num2hobby = {}
        self.scores = []
        for idx, key in enumerate(self.datadict_from_json):
            self.num2hobby[idx] = key
            self.scores.append(self.datadict_from_json[key])
        
        self.position_score = []
        self.score_bias = []

    def setBias(self, position_score):
        self.position_score = position_score
        
        for idx in range(len(position_score)):
            if idx == 0: 
                self.score_bias.append(0)
            else:
                self.score_bias.append(sum(position_score[:idx]))
    
    def getDatasetWithBias(self):
        scores_with_bias = []
        for score in self.scores:
            score_to_data = [(a+b-1) for a, b in zip(score, self.score_bias)]
            scores_with_bias.append(score_to_data)

        return scores_with_bias

    def getDataWithBias(self, score):
        return [(a+b-1) for a, b in zip(score, self.score_bias)]

    def getNum2Hobby(self):
        return self.num2hobby

    def getLen(self):
        return len(self.scores[0])

    def getCount(self):
        return len(self.scores)