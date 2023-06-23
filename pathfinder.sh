#!/bin/bash

relation=$1
python sl_policy.py $relation
python policy_agent.py $relation retrain
python policy_agent.py $relation test


echo 按任意键继续
read -n 1
echo 继续运行

