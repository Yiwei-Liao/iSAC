# Reasoning over the Air: A Reasoning-based Implicit Semantic Communication Framework
Generative adversarial imitation learning-based solution(G-RML)
Different from traditional communication solutions, the source encoder of the proposed G-RML does not focus only on sending as much of the useful messages as possible; but, instead, it tries to guide the destination user to learn a reasoning mechanism to map any observed explicit semantics to the corresponding implicit semantics characterized by a set of possible reasoning paths involving the hidden entities and relations that are most relevant to the semantic meaning. By applying G-RML, we prove that the destination user can accurately imitate the reasoning process of the source user and automatically generate the reasoning paths following the same probability distribution as the expert paths. Compared to the traditional semantic communication solutions, our proposed G-RML requires much less communication and computational resources and scales well to the scenarios involving the communication of complex semantic meanings with a large number of concepts and relations.
## How to use the code
#### 0. install the environment
```shell
pip install -r requirements.txt
```
#### 1. download the dataset and unzip 
Please download the datasets in this repository [DeepPath](https://github.com/xwhan/DeepPath/tree/master)
#### 2. to train the model
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

#### 4. to train the embeddings
```shell
# run
python train_transe_FB15K237 --Dim 200 --GPU 1
```
## If you use our code, please cite the paper
```
```
## Ackonwledgement
- [DeepPath](https://github.com/xwhan/DeepPath/tree/master)
- [OpenKE](https://github.com/thunlp/OpenKE)