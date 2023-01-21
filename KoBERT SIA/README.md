# KoBERT-nsmc

- KoBERT를 이용한 네이버 영화 리뷰 감정 분석 (sentiment classification)
- 🤗`Huggingface Tranformers`🤗 라이브러리를 이용하여 구현

## Dependencies

- torch==1.4.0
- transformers==2.10.0
- sentencepiece==0.1.97

## Train

```bash
$ python3 main.py --model_type kobert --do_train --do_eval
```

## Prediction

 - [여기](https://drive.google.com/drive/folders/1-83lNwn58RE9bVOKKJ_h0uMzvrx587AU?usp=sharing)서 pre-trained 파일들을 다운 받아 ```./model``` 디렉토리에 넣어줍니다.


```bash
$ python3 predict.py --input_file {INPUT_FILE_PATH} --output_file {OUTPUT_FILE_PATH} --model_dir {SAVED_CKPT_PATH}
```
default ```INPUT_FILE_PATH``` : ```./sample_pred_in.txt```

default ```OUTPUT_FILE_PATH``` : ```./sample_pred_out.txt```

default ```SAVED_CKPT_PATH``` : ```./model```

## Results

|                   | Accuracy (%) |
| ----------------- | ------------ |
| KoBERT            | **89.63**    |
| DistilKoBERT      | 88.41        |
| Bert-Multilingual | 87.07        |
| FastText          | 85.50        |

## References

- [KoBERT](https://github.com/SKTBrain/KoBERT)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [NSMC dataset](https://github.com/e9t/nsmc)

## Acknowledgments
* [원작자](https://github.com/monologg)분께 코드 공유와 관련하여 감사의 말씀 전합니다 :)
