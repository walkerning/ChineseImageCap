TODO
-----
- [x] 现在的结果... 不管是哪个集上, 不管是哪个图片...第一个词一定是一个. 第二个词基本都是"男人"或者"人". 然后后面都是"在". 然后有"玩滑板", "滑雪", "冲浪"...完全不对啊! 虽然"滑板"这种东西没有detection也不相信能准确检测出来...但是现在的问题应该是因为这样的词组合出现频率太多了... 完全在猜哪个词最可能出现, 在学频率.... 把image数据的影响完全覆盖掉了....第一个词一定是一个这种= =想想啊... 调整loss对每个词的错判loss用TF-IDF 加权?为什么NIC本身文章里面没有这种东西...妈的...发现是DOS的utf8格式文件的BOM没处理好导致的....
- [x] 写一个简单的把phrase的embedding拿出来测试的脚本, 和一些简单的query. 看看训出来的embedding怎么样
- [x] 不cut试一次. check the quality of word segmentation. is there any improvement methods? **最后评价应该是不切割... 所以用nocut比较好**
- [ ] try training word embedding seperately. or use other corpus to train word embedding
- [ ] 卧槽... 是都没代码吗... try detection-based method: scanning-detection-based continous attribute vector: "What Value Do Explicit High Level Concepts have in Vision to Language Problems". MIL-based discrete attribute + MELM: "From Captions to Visual Concepts and Back"
- [ ] 卧槽... 是没代码吗??? try semantic attention method: "Image Captioning with Semantic Attention"
- [ ] try Attention method: "show, attend, and tell"

metric evaluation
----

在train/val上eval metric, 也需要跑`run_inference.py`, 指定`--dataset_name train_set/validation_set`, 也需要提供相应的`test.txt`里包含index. (0-7999或者0-999)
```
CUDA_VISIBLE_DEVICES="3" python run_inference.py --checkpoint_path=./trainlog/model.ckpt-5000 --vocab_file=../../vocab.txt --feature_files ../../../data/image_vgg19_fc1_feature_78660340.h5 --test_index_file=test.txt --write_top_res_file train_candidate.txt --dataset_name train_set
```

- [x] BLEU1~4: `python eval/eval.py ../../train.txt train_candidate.txt test.txt`
- [x] 多个ROUGE版本
- [x] METEOR
- [x] CIDEr
- [ ] 想有分词性的evaluation. 需要在之前加一个HMM/CRF对vocab里每个词标词性

end-to-end训练之后的embedding质量
----

`python evaluate_embedding.py ../../vocab.txt trainlog_fc1/model.ckpt-100000 embedding_trainlog_fc1_100000.pkl`

* syntactic relation: 中文syntactic的relation感觉没啥好看的...因为不太清楚怎么定义这里的syntactic relation...
* semantic relation: 对于出现的比较多的词(大概大于几十次)找到的最接近的词基本词性和词义的确相近. 对于出现的很少的词就是乱的... 词之间的relation的结果并不好...
* l2 norm: 出现次数越多的phrase l2norm一般比较大. 不过还没有明确看出有什么很特别的...这个在思考一下. 有没必要对这个embedding做一些工作.

现在测试的结果存在`embeddingtest_fc1_100000.log`中.



512维的embedding和hidden state size基本没差别...
* fc1 nocut 300: INFO:tensorflow:global step 100000: loss = 2.3292 (0.04 sec/step)
* fc1 nocut 512: INFO:tensorflow:global step 100000: loss = 2.0021 (0.08 sec/step)
* fc1 512: INFO:tensorflow:global step 500000: loss = 1.9773 (0.03 sec/step)
* fc1 fc2 512: INFO:tensorflow:global step 500000: loss = 2.0438 (0.05 sec/step)
* fc1 fc2 nodropout 512: INFO:tensorflow:global step 500000: loss = 1.9051 (0.03 sec/step)

* fc1 nocut 512 100k-iter. val:
Bleu_1: 0.640440928994
Bleu_2: 0.51104388127
Bleu_3: 0.402453928653
Bleu_4: 0.317171016579
METEOR: 0.23773381406
ROUGE_L: 0.496505258212
CIDEr: 1.0176400948
------
* fc1 nocut 512 400k-iter. val:
Bleu_1: 0.682810149694
Bleu_2: 0.555341377574
Bleu_3: 0.44442600632
Bleu_4: 0.353280266137
METEOR: 0.256156386396
ROUGE_L: 0.530428274948
CIDEr: 1.19092864445