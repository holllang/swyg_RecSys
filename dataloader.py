import json

class DataLoader:
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf8') as f:
            self.datadict_from_json =  json.load(f)
        
        self.num2hobby = {}
        self.answers = []
        for idx, key in enumerate(self.datadict_from_json):
            self.num2hobby[idx] = key
            self.answers.append(self.datadict_from_json[key])
        
        self.num_per_question = []
        self.question_bias = []

    def setBias(self, num_per_question):
        self.num_per_question = num_per_question
        
        for idx in range(len(num_per_question)):
            if idx == 0: 
                self.question_bias.append(0)
            else:
                self.question_bias.append(sum(num_per_question[:idx]))
    
    def getDatasetWithBias(self):
        answers_with_bias = []
        for answer in self.answers:
            answer_to_data = [(a+b-1) for a, b in zip(answer, self.question_bias)]
            answers_with_bias.append(answer_to_data)

        return answers_with_bias

    def getDataWithBias(self, answer):
        return [(a+b-1) for a, b in zip(answer, self.question_bias)]

    def getNum2Hobby(self):
        return self.num2hobby

    def getLen(self):
        return len(self.answers[0])

    def getCount(self):
        return len(self.answers)