# TegFormer
Code for EMNLP 2022 submission titled "TegFormer: Towards Better Topic-Consistency and Text-Logic on Topic-to-Essay Generation"

## --Instructions--  
### For Essay  
Step1: Download data from https://pan.baidu.com/s/1_JPh5-g2rry2QmbjQ3pZ6w (Essay)  
Step2: Download gpt2 from https://huggingface.co/uer/gpt2-chinese-cluecorpussmall and put downloaded files into TegFromer/Essay/model/gpt2/   
Step3: cd TegFormer/Essay/data,   run TextRank.py and  then run Extend_topics.py  
Step4: cd TegFormer/Essay, run train.py  
Step5: put test.txt  into TegFormer/Essay/data   
Step6: cd TegFormer/Essay, run infer.py  and you can get generated text from TegFormer/Essay/data/gen.txt  


### For Zhihu  
Step1: Download data from https://pan.baidu.com/s/1eC4gb_We33kr-ZbHn3KdIA   
Step2: Download gpt2 from https://huggingface.co/uer/gpt2-chinese-cluecorpussmall and put downloaded files into TegFromer/Zhihu/model/gpt2/  
Step3: cd TegFormer/Zhihu/data,   run TextRank.py and  then run Extend_topics.py  
Step4: cd TegFormer/Zhihu, run train.py  
Step5: put test.txt  into TegFormer/Essay/Zhihu  
Step6: cd TegFormer/Zhihu, run infer.py  and you can get generated text from TegFormer/Zhihu/data/gen.txt  

