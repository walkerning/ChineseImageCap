CUDA_VISIBLE_DEVICES="3" python run_inference.py --checkpoint_path=./trainlog/model.ckpt-5000 --vocab_file=../../vocab.txt --feature_files ../../../data/image_vgg19_fc1_feature_78660340.h5 --test_index_file=test.txt

TODO
-----
- [ ] 现在的结果... 不管是哪个集上, 不管是哪个图片...第一个词一定是一个. 第二个词基本都是"男人"或者"人". 然后后面都是"在". 然后有"玩滑板", "滑雪", "冲浪"...完全不对啊! 虽然"滑板"这种东西没有detection也不相信能准确检测出来...但是现在的问题应该是因为这样的词组合出现频率太多了... 完全在猜哪个词最可能出现, 在学频率.... 把image数据的影响完全覆盖掉了....第一个词一定是一个这种= =想想啊... 调整loss对每个词的错判loss用TF-IDF 加权?为什么NIC本身文章里面没有这种东西...
- [ ] try training word embedding seperately. or use other corpus to train word embedding
- [ ] check the quality of word segmentation. is there any improvement methods?
- [ ] 卧槽... 是都没代码吗... try detection-based method: scanning-detection-based continous attribute vector: "What Value Do Explicit High Level Concepts have in Vision to Language Problems". MIL-based discrete attribute + MELM: "From Captions to Visual Concepts and Back"
- [ ] 卧槽... 是没代码吗??? try semantic attention method: "Image Captioning with Semantic Attention"
- [ ] try Attention method: "show, attend, and tell"


