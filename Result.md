## XLM-Roberta类模型
使用2.5TB过滤的CommonCrawl数据训练，包含了100种语言。
在XLU（跨语言理解）上取得了SOTA。

xlmr-base：使用bert-base的架构，有250M参数。
xlmr-large：使用bert-large的架构，有560M参数，2倍xlmr-base。
xlmr-xl：layers=36, model_dim=2560，有3.5B参数，10倍xlmr-base。
xlmr-xxl：layers=48，model_dim=4096，有10.7B参数，30倍xlmr-base。
用的都是同一套分词器。

### xlm-roberta-base类
Records/cardiffnlp/twitter-xlm-roberta-base-sentiment
    En2En   En2Es   En2Fr
0  78.818  71.862  67.244
1   0.554   1.605   1.503
2   6.000   6.000   6.000

Records/CodeNinja1126/xlm-roberta-large-kor-mrc
    En2En   En2Es   En2Fr
0  82.893  77.793  73.931
1   0.871   0.305   0.965
2   3.000   3.000   3.000

Records/xlm-roberta-base
    En2En   En2Es   En2Fr
0  80.199  74.029  69.123
1   2.060   1.060   1.457
2   3.000   3.000   3.000

Records/xlm-roberta-base-yelp-mlm
    En2En   En2Es   En2Fr
0  81.915  70.275  68.742
1   0.243   1.115   2.589
2   3.000   3.000   3.000

# xlm-roberta-large类
Records/xlm-roberta-large
    En2En   En2Es   En2Fr
0  81.845  76.662  73.922
1   0.122   0.995   0.715
2   3.000   3.000   3.000

Records/xlm-roberta-large-finetuned-conll02-spanish
    En2En   En2Es   En2Fr
0  83.593  75.968  73.206
1   0.703   1.694   0.399
2   3.000   3.000   3.000

Records/xlm-roberta-large-finetuned-conll03-english
    En2En   En2Es   En2Fr
0  82.469  77.002  73.523
1   0.973   1.113   1.056
2   3.000   3.000   3.000


## BERT类
24 smaller BERT models，只使用英语训练。效果也挺好。

提出了whole word masking models。

bert-large 14层，1024-hidden，16-heads，340M参数。

使用whole word masking技术，效果比原始随机mask有提升.

BERT-Base，multilingual cased：104种语言，12层，768-hidden，12个head，110M参数。

BERT-Base，Chinese

### mBert类

Records/bert-base-multilingual-uncased
    En2En   En2Es   En2Fr
0  77.113  67.652  62.266
1   0.108   0.864   0.716
2   3.000   3.000   3.000

Records/dbmdz/bert-base-multilingual-cased-finetuned-conll03-spanish
    En2En   En2Es   En2Fr
0  75.722  65.755  62.417
1   1.144   2.528   1.407
2   3.000   3.000   3.000

Records/nlptown/bert-base-multilingual-uncased-sentiment
    En2En   En2Es   En2Fr
0  76.131  67.721  62.293
1   1.343   2.066   2.206
2   7.000   7.000   7.000

Records/google/rembert
    En2En   En2Es   En2Fr
0  81.092  76.544  73.413
1   0.635   0.473   0.514
2   3.000   3.000   3.000


### bert-base类
Records/activebus/BERT_Review
    En2En   En2Es   En2Fr
0  81.839  10.122  33.155
1   0.600   1.926   2.137
2   3.000   3.000   3.000

Records/activebus/BERT-XD_Review
    En2En   En2Es   En2Fr
0  80.277  19.476  39.780
1   0.242  10.392   2.884
2   3.000   3.000   3.000

Records/ainize/klue-bert-base-mrc
    En2En  En2Es   En2Fr
0  71.171  9.402  15.881
1   0.200  1.464   2.686
2   3.000  3.000   3.000

Records/bert-base-uncased
    En2En  En2Es  En2Fr
0  77.765  0.902  15.95
1   1.027  0.883   2.65
2   3.000  3.000   3.00

Records/skimai/spanberta-base-cased-ner-conll02
    En2En   En2Es   En2Fr
0  68.608  42.478  28.412
1   2.193  24.963   1.251
2   3.000   3.000   3.000

Records/SpanBERT/spanbert-base-cased
    En2En  En2Es   En2Fr
0  75.627  4.028  16.463
1   0.994  2.350   3.116
2   3.000  3.000   3.000

Records/Tahsin/BERT-finetuned-conll2003-POS
    En2En  En2Es   En2Fr
0  77.471  8.455  16.677
1   0.642  3.863   1.052
2   3.000  3.000   3.000


### bert-large类
Records/bert-large-uncased
    En2En   En2Es   En2Fr
0  78.571  16.101  28.883
1   0.973   8.591   6.349
2   3.000   3.000   3.000

## roberta-base
只用MLM训练
使用了BookCorpus，英语维基百科，CC-News，OpenWebText,Stories数据训练。

### roberta-large类
Records/roberta-large
    En2En   En2Es   En2Fr
0  81.269  64.335  56.028
1   1.518   2.776   4.131
2   3.000   3.000   3.000

Records/this-is-real/mrc-pretrained-roberta-large-1
    En2En   En2Es   En2Fr
0  70.529  17.274  18.594
1   0.000   0.000   0.000
2   1.000   1.000   1.000

### albert类
Records/albert-base-v2
    En2En  En2Es   En2Fr
0  78.604  3.838  21.421
1   1.186  1.669   6.987
2   5.000  5.000   5.000


# albert类
Records/albert-base-v2
    En2En  En2Es   En2Fr
0  78.604  3.838  21.421
1   1.186  1.669   6.987
2   5.000  5.000   5.000

# electra-base类
Records/dbmdz/electra-base-french-europeana-cased-generator
    En2En   En2Es   En2Fr
0  63.907  19.191  45.311
1   1.778   1.688   1.290
2   3.000   3.000   3.000

Records/electra-base-discriminator-yelp-mlm
    En2En  En2Es   En2Fr
0  80.817  7.799  34.348
1   0.831  7.949   5.876
2   3.000  3.000   3.000

Records/google/electra-base-discriminator
    En2En  En2Es   En2Fr
0  80.558  8.084  27.262
1   1.088  5.154   8.196
2   3.000  3.000   3.000

# electra-small类
Records/google/electra-small-discriminator
    En2En  En2Es   En2Fr
0  76.649  2.943  16.686
1   1.314  1.504   2.620
2   3.000  3.000   3.000

Records/test-electra-small-yelp
    En2En  En2Es   En2Fr
0  79.137  6.270  24.254
1   0.333  2.229   1.056
2   3.000  3.000   3.000
