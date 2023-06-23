import argparse
import pandas as pd
import os
import time
import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--Dim", type=int)
parser.add_argument("--GPU", type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=args.GPU # 改GPU编号

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/FB15K237/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/FB15K237/", "link")

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = args.Dim, 
	p_norm = 1, 
	norm_flag = True)


# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size()
)

ResultList = list()
TimeList = list()
init = time.time()
for i in range(10):
	st = time.time()
	trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 100, alpha = 1.0, use_gpu = True)
	trainer.run()
	dur = time.time() - st
	print("Training time of 10 Rounds : ", dur)
	tester = Tester(model=transe, data_loader=test_dataloader, use_gpu=True)
	result = tester.run_link_prediction(type_constrain=False)
	print("result: ", result)
	TimeList.append(dur)
	ResultList.append(result)

filename1 = "./resultRecord/TransE_FB15k237_Result_Dim" + str(transe.dim) + ".csv"
filename2 = "./resultRecord/TransE_FB15k237_Time_Dim" + str(transe.dim) + ".csv"
df1 = pd.DataFrame(data=ResultList)
df1.to_csv(filename1, encoding='utf-8', index=False)
df2 = pd.DataFrame(data=TimeList)
df2.to_csv(filename2, encoding='utf-8', index=False)