# TegFormer  
Code for ArXiv submission titled "TegFormer: Towards Better Topic-Consistency and Text-Logic on Topic-to-Essay Generation"  (https://arxiv.org/abs/2212.13456)  

## --Field--
CS Concepts: Natural Language Processing -->  Natural Language Generation --> Topic-to-essay Generation  
![1672910142129](https://user-images.githubusercontent.com/29347148/210744194-4d7ceae3-a4df-4887-b5ae-39027f72f544.png)  

## --Net Architecture--  
![1672909915852](https://user-images.githubusercontent.com/29347148/210743393-4187cf22-30c4-4e58-bdfe-9573d59eb1d4.png)  

## --Experiments--   
### Main experiment  
![1672909978501](https://user-images.githubusercontent.com/29347148/210743621-30448262-de31-408e-8d0a-37718f8f7113.png)  
### Hyperparameter Analysis
![1672910058032](https://user-images.githubusercontent.com/29347148/210743881-7f831287-bea3-44c2-87ac-8a12816b2420.png)  
### Ablation Studies
![1672910117930](https://user-images.githubusercontent.com/29347148/210744079-ae606329-f26b-4b69-aa20-9ea54bf5ccb7.png)  


## --Instructions--  
### For Essay  
Step1: Download data from https://pan.baidu.com/s/1_JPh5-g2rry2QmbjQ3pZ6w   
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

