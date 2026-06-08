import torch, os, time, pickle
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from Bio import SeqIO
import itertools as it
torch.manual_seed(0)
width =  os.get_terminal_size().columns-30

class fc(nn.Module):
    def __init__(self):
        super(fc, self).__init__()
        self.fc1 = nn.Linear(1280,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 16)
        self.fc5 = nn.Linear(16, 1)
        self.sig = nn.Sigmoid()
    def forward(self, input_sentence):
        embedding0 = input_sentence[:, 0, :].reshape(-1,1280)
        embedding1 = input_sentence[:, 1, :].reshape(-1,1280)
        x = torch.mul(embedding0.reshape(-1,1280), embedding1.reshape(-1,1280))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.sig(self.fc5(x))
        return x

with open("embedding.pkl","rb") as f:
	esm_embed = pickle.load(f)
idPairTrain = np.loadtxt('c1Train.txt', dtype = str, usecols = (0, 1, 2))


trainId = []
trainLabel = []
for i, j, l in idPairTrain:
    trainId.append([i, j])
    trainLabel.append(int(l))
trainIdArray = np.array(trainId)
trainLabelArray = np.array(trainLabel)


Coding = []
for i, j in trainIdArray:
    Coding.append(torch.stack([esm_embed[str(i)], esm_embed[str(j)]], dim = 0))

Xtrain = torch.stack(Coding, dim = 0)
ytrain = torch.tensor(trainLabelArray, dtype=torch.long)   
trainDataset = Data.TensorDataset(Xtrain, ytrain)
trainLoader = Data.DataLoader(dataset=trainDataset, batch_size=128, shuffle=True, num_workers=0)

EPOCH = 40
LR = 0.001

lstms = fc()
lstms.cuda()
optimizer = torch.optim.Adam(lstms.parameters(), lr=LR)
loss_func = nn.BCELoss()
loss_func.cuda()


print('Training Starts')

start2 = time.time()
for epoch in range(1,EPOCH+1):
    start = time.time()
    for step, (batch_x, batch_y) in enumerate(trainLoader):        
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()        
        output = lstms(batch_x)      
        loss = loss_func(torch.flatten(output).float(), batch_y.cuda().float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end = time.time()
        tmp = len(trainLoader)/width
        if step % tmp < 1:
            print(f'\rEpoch:{epoch} [{"-"*(int(step/tmp) + 1)}->{" "*(width - 1 -int(step/tmp))}][{(end-start):.1f}s][{(end-start2):.0f}s]  ', end='')
print()

seqs = {i.id:str(i.seq) for i in SeqIO.parse('seqs.fasta', 'fasta')}
idAll = list(seqs.keys())
pairAll = list(it.combinations(idAll, 2))
pairAll.extend(pairAll[0:10])



with open('tAll.txt', 'w') as f:
    start = time.time()
    with torch.no_grad(): 
        lstms.train(False)
        for i in range(2723222):
            XtestBatchId = []
            XtestBatchCoding = []
            for XtestBatch in pairAll[128*(i):128*(i+1)]:   
                XtestBatchId.append([XtestBatch[0], XtestBatch[1]])
                XtestBatchCoding.append(torch.stack([esm_embed[XtestBatch[0]], esm_embed[XtestBatch[1]]], dim = 0))

            XtestBatchCoding = torch.stack(XtestBatchCoding, dim = 0)    
            output = lstms(XtestBatchCoding.cuda()).tolist()
            
            for j in range(128):

                f.write(f'{XtestBatchId[j][0]}\t{XtestBatchId[j][1]}\t{output[j][0]:0.8f}\n')
            print(f'\r{i}/2723222 batches | {(i/2723222*100):.2f}% | {(time.time()-start):.1f}s', end="")                                  
