import numpy as np
import torch
import numpy.random as random
from load_data import Data
from main import Experiment

class noiseAdd():
    '''
    Parameters:
        mode: Number of bits used for PSK modulation

        snr: Signal-to-noise ratio (unit: dB)
    '''
    def __init__(self,mode=8,snr=1):
        # self.symbol=parameter
        # self.symbol
        # self.row=self.symbol.shape[0]
        # self.col=self.symbol.shape[1]
        self.mode=mode
        self.snr=snr

    def floatToBinary(self,k):
        bits=np.zeros(self.mode)
        if(k<0):
            bits[0]=1
            k=-k
        for i in range(1,self.mode):
            k=k*2
            bits[i]=k//1
            k=k-k//1
        return bits

    def BPSK(self, parameter):
        self.symbol=parameter
        self.row = self.symbol.shape[0]
        self.col = self.symbol.shape[1]

        newParameter=np.zeros(0)
        for i in range(0,self.row):
            # print("Noise Adding Index:", i)
            # print("Initial Symbol: ", self.symbol[i])
            rowVector=np.zeros(0)
            for j in range(0,self.col):
                rowVector=np.append(rowVector, self.floatToBinary(self.symbol[i][j]))    # 对每个数进行二进制化
            # 得到行向量二进制化后的结果
            for k in range(0,self.mode*self.col):
                if rowVector[k]==0:
                    rowVector[k]=-1
            # print("Binary Vector: ", rowVector)
            channelSig=self.channel(rowVector)
            Rx=self.recover(channelSig)
            # print("Rx:", Rx)
            newParameter=np.append(newParameter,Rx,axis=0)
            # print("New Parameter: ", newParameter)
        newParameter=newParameter.reshape(self.row, self.col)
        # print("New Parameter: ", newParameter)

        ## File Storage
        # np.savetxt("ParametersAfterNoiseAddition.txt", newParameter, delimiter=',')
        return newParameter

    def recover(self,signal):
        for i in range(0,self.col*self.mode):
            if signal[i]>0:
                signal[i]=1
            else:
                signal[i]=0
        symbols=np.zeros(0)
        for i in range(0,self.col):
            symbol=0
            sign=1
            multiplier=0.5
            for j in range(0,self.mode):
                if j==0 and signal[8*i+j]==1:
                    sign=-1
                if j>0:
                    symbol=symbol+signal[8*i+j]*multiplier
                    multiplier=multiplier*0.5
            symbol=symbol*sign
            symbols=np.append(symbols,symbol)
        # print("Symbols: ", symbols)
        return symbols

    def channel(self, signal):
        # 高斯加性白噪声
        noise = random.randn(self.col*self.mode)
        p = 10 ** (self.snr / 10)
        an = 1 / np.sqrt(2 * p)
        noise = an * noise
        # 瑞利衰落
        h = np.random.rayleigh(size=signal.shape)
        signal = h * signal
        # print("Signal:", signal)
        # print("Noise:", noise)
        finalSignal=signal+noise
        # print("After Adding Noise: ", finalSignal)
        return finalSignal


    def __init__(self, dataset, model):
        self.dataset = dataset
        self.d = Data(data_dir="data/%s/" % self.dataset)
        self.model=model
        self.cuda = False

    def evaluateModel(self):
        # print("Noise:", self.noise)
        self.train_and_eval()
        self.evaluate(self.d.test_data)
        # self.evaluate()

    def train_and_eval(self):
        self.entity_idxs = {self.d.entities[i]: i for i in range(len(self.d.entities))}
        self.relation_idxs = {self.d.relations[i]: i for i in range(len(self.d.relations))}

    def evaluate(self,data):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        sr_vocab = self.get_er_vocab(self.get_data_idxs(self.d.data))

        print("Number of data points: %d" % len(test_data_idxs))

        for i in range(0, len(test_data_idxs)):
            data_point = test_data_idxs[i]
            e1_idx = torch.tensor(data_point[0])
            r_idx = torch.tensor(data_point[1])
            e2_idx = torch.tensor(data_point[2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            predictions_s = self.model.forward(e1_idx.repeat(len(self.d.entities)),
                                          r_idx.repeat(len(self.d.entities)), range(len(self.d.entities)))

            filt = sr_vocab[(data_point[0], data_point[1])]
            target_value = predictions_s[e2_idx].item()
            predictions_s[filt] = -np.Inf
            predictions_s[e1_idx] = -np.Inf
            predictions_s[e2_idx] = target_value

            sort_values, sort_idxs = torch.sort(predictions_s, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()
            rank = np.where(sort_idxs == e2_idx.item())[0][0]
            ranks.append(rank + 1)

            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)

        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))

if __name__=="__main__":
    ## Parameter Definition
    dataset="NELL-995"
    dimension=40
    SNR=50
    PSK=8

    filePath_model = "./" + dataset + "/model_" + dataset + "_"+ str(dimension) + "d.pkl"

    model = torch.load(filePath_model, map_location="cpu")
    EntityEmbeddings=model.Eh.weight
    RelationEmbeddings=model.rvh.weight
    print("Initial Entity: ", EntityEmbeddings)
    print("Initial Relation: ", RelationEmbeddings)
    nA=noiseAdd(PSK,SNR)
    newEntity=nA.BPSK(EntityEmbeddings)
    newRelation=nA.BPSK(RelationEmbeddings)

    print("Noisy Entity: ", newEntity)
    print("Noisy Relation: ", newRelation)

    newEntity=torch.from_numpy(newEntity)
    newEntity=torch.nn.Parameter(newEntity)

    newRelation = torch.from_numpy(newRelation)
    newRelation = torch.nn.Parameter(newRelation)

    with torch.no_grad():
        model.Eh.weight=newEntity
        model.rvh.weight=newRelation
    # print("New Model: ", model1.Eh.weight)