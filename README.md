# README
Code and Datasets for "Learning Fine-Grained Fact-Article Correspondence in Legal Cases". For details of the model and experiments, please see our [paper](https://ieeexplore.ieee.org/document/9627791).

## Pretrained Wordvecs
The word vectors used in our experiments:
- embedding_matrix.pkl: https://1drv.ms/u/s!AoHUnvdb_8b2h03KopN3j5LNtOGS?e=ZLkgMT
- vocab.pkl: https://1drv.ms/u/s!AoHUnvdb_8b2h0zUab8JbJPatPZD?e=9uuUGd

Download and save them in './data/'

## Example

(All examples below take the crime of intentionally injuring as example)

- Train MLMN for law article recommendation:
    ```text
    python main.py --crime hurt --negtive_multiple 5 
    ```
  
- Train MLMN with parsed infomation
    ```text
    python main.py --model_name ThreeLayersWithElement --crime hurt --negtive_multiple 5
    ```

- Train MLMN with RoBERTa
    ```text
    python main.py --use_pretrain_model --model_name ThreeLayersPretrain --crime hurt --negtive_multiple 5 --embedding_dim 768 --batch_size 16 --epochs 10 --earlystop_patience 3
    ```

- Train model for penalty prediction with fine-grained fact-article correspondence
    ```text
    python decision_predictor.py --fine_grained_penalty_predictor --crime hurt
    ```

- Train model for penalty prediction with coarse-grained fact-article correspondence
    ```text
    python decision_predictor.py --crime hurt 
    ```

## Citation

```text
@ARTICLE{ge2021learning,
    author={Ge, Jidong and Huang, Yunyun and Shen, Xiaoyu and Li, Chuanyi and Hu, Wei},
    journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
    title={Learning Fine-Grained Fact-Article Correspondence in Legal Cases}, 
    year={2021},
    volume={29},
    pages={3694-3706},
    doi={10.1109/TASLP.2021.3130992}
}
```
