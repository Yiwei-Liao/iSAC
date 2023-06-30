import argparse
import pandas as pd
import os
import time
import source
from source.config import Trainer, Tester
from source.module.model import TransE
from source.module.loss import MarginLoss
from source.module.strategy import NegativeSampling
from source.data import TrainDataLoader, TestDataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int)
parser.add_argument("--gpu", type=str)
parser.add_argument("--dataset",type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=args.GPU
file_path = "./benchmarks/" + args.dataset + "/"

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = file_path, 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader(file_path, "link")

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

# filename1 = "./resultRecord/TransE_FB15k237_Result_Dim" + str(transe.dim) + ".csv"
# filename2 = "./resultRecord/TransE_FB15k237_Time_Dim" + str(transe.dim) + ".csv"
# df1 = pd.DataFrame(data=ResultList)
# df1.to_csv(filename1, encoding='utf-8', index=False)
# df2 = pd.DataFrame(data=TimeList)
# df2.to_csv(filename2, encoding='utf-8', index=False)