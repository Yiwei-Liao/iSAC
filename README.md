# Reasoning over the Air: A Reasoning-based Implicit Semantic Communication Framework
The Generative Adversarial Imitation Learning-based Solution (G-RML) represents a paradigmatic shift in semantic communication by introducing a novel source encoder. Unlike traditional models that focus on maximizing the transmission of explicit semantic messages, G-RML aims to facilitate a reasoning mechanism at the destination user's end. This mechanism enables the mapping of observed explicit semantics to their corresponding implicit semantics, guided by hidden entities and relationships. The primary objective is to ensure that the destination user can accurately imitate the source user's reasoning process, generating paths that follow the same probability distribution as expert paths. Employing Generative Adversarial Imitation Learning techniques, G-RML operates with reduced computational and communication resources and scales efficiently to complex semantic scenarios. Experimental validation substantiates its effectiveness, rendering it superior in resource utilization and semantic versatility compared to traditional solutions. This advancement positions G-RML as a groundbreaking approach in the field of semantic communication, with implications for future research and potential standardization in emerging 6G technologies.


## How to use the code
#### 0. install the environment
```shell
pip install -r requirements.txt
```
#### 1. download the dataset for Inference
download the dataset and unzip in ./SemanticInference
[FB15K237](https://drive.google.com/file/d/1klWL11nW3ZS6b2MtLW0MHnXu-XlJqDyA/view) [NELL-995](https://paperswithcode.com/dataset/nell-995)
#### 2. to train the Semantic Encoder
in ./SemanticEncoder
```shell
# run
python train_transe --dim 200 --gpu 1 --dataset FB15K237
```

#### 2. to train the Semantic Comparator and Interpreter
in ./SemanticInference
```shell
# run
python sl_policy.py $relation
python policy_agent.py $relation retrain
python policy_agent.py $relation test
# or you can run the shell script
./pathfinder.sh ${relation_name}
```
#### 3. to test the model
```shell
# run 
python fact_prediction_eval.py $relation
./link_prediction_eval.sh ${relation_name}
```

## If you use our code, please cite the paper
```
Yong Xiao, Yiwei Liao, Yingyu Li, Guangming Shi, H. Vincent Poor, Walid Saad, Merouane Debbah, and Mehdi Bennis, "Reasoning over the Air: A Reasoning-based Implicit Semantic-Aware Communication Framework," accepted at IEEE Transactions on Wireless Communications (TWC).
url: https://ieeexplore.ieee.org/document/10250170
```
