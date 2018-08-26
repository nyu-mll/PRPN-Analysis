# PRPN Analysis

This repo contains the output files and analysis results reported in the paper **"Grammar Induction with Neural Language Models: An Unusual Replication"** [1], where we perform an in-depth analysis of the **Parsing Reading Predict Networks** [2].

The parsed files can be downloaded [here.](https://drive.google.com/file/d/1Zc6lgqyohCcNKlqp6gk_J0RMCbCS1gHR/view?usp=sharing)
The parsed files are named in the following way:
- parsed_{parsed-dataset}_{model-type}_{train-data}_{earlystop-criterion}.jsonl
- Example: parsed_WSJ_PRPNUP_WSJFull_ESUP.jsonl

We also share the pretrained model that provides the best F-1 score (PRPN-LM trained on AllNLI with language modeling criterion) which can be downloaded [here](https://drive.google.com/file/d/1BHW9Gd1ackTVZfG3ZIXw5KupFRc8dvHH/view?usp=sharing).

### References
[1] Phu Mon Htut, Kyunghyun Cho, Samuel R. Bowman. ***Grammar Induction with Neural Language Models: An Unusual Replication***. To appear in Proceedings of the EMNLP. 2018.

[2] Yikang Shen, Zhouhan Lin, Chin wei Huang, and Aaron Courville. [***Neural language modeling by jointly learning syntax and lexicon***](https://arxiv.org/abs/1711.02013). Proceedings  International Conference on Learning Representations. 2018. [[code](https://github.com/yikangshen/PRPN)]

