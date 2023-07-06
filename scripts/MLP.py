import torch, os, time, pickle
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
torch.manual_seed(0)
width =  os.get_terminal_size().columns-30

# MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(1280,1024)
        self.l2 = nn.Linear(1024,512)
        self.l3 = nn.Linear(512, 128)
        self.l4 = nn.Linear(128, 16)
        self.l5 = nn.Linear(16, 1)
        self.sig = nn.Sigmoid()
    def forward(self, input_sentence):
        embedding0 = input_sentence[:, 0, :].reshape(-1,1280)
        embedding1 = input_sentence[:, 1, :].reshape(-1,1280)
        x = torch.mul(embedding0.reshape(-1,1280), embedding1.reshape(-1,1280))
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = self.sig(self.l5(x))
        return x

# Load ESM embedding
with open("features/esm1b_t33_650M_UR50S.pkl","rb") as f:
	esm_embed = pickle.load(f)

# Data preprocessing
idPairTrain = np.loadtxt('datasets/c1Train.txt', dtype = str, usecols = (0, 1, 2))
idPairc2 = np.loadtxt('datasets/c2Test.txt', dtype = str, usecols = (0, 1, 2))
idPairc3 = np.loadtxt('datasets/c3Test.txt', dtype = str, usecols = (0, 1, 2))

# Train dataset
trainId = []
trainLabel = []
for i, j, l in idPairTrain:
    trainId.append([i, j])
    trainLabel.append(int(l))
trainIdArray = np.array(trainId)
trainLabelArray = np.array(trainLabel)

embedding = []
for i, j in trainIdArray:
    embedding.append(torch.stack([esm_embed[i], esm_embed[j]], dim = 0))

Xtrain = torch.stack(embedding, dim = 0)
ytrain = torch.tensor(trainLabelArray, dtype=torch.long)   
trainDataset = Data.TensorDataset(Xtrain, ytrain)
trainLoader = Data.DataLoader(dataset=trainDataset, batch_size=128, shuffle=True, num_workers=0)

# C2 Test dataset
c2Id = []
c2Label = []
for i, j, l in idPairc2:
    c2Id.append([i, j])
    c2Label.append(int(l))
c2Id = np.array(c2Id)
c2Label = np.array(c2Label)

c2Emdedding = []
for i, j in c2Id:
    c2Emdedding.append(torch.stack([esm_embed[i], esm_embed[j]], dim = 0))
c2Emdedding = torch.stack(c2Emdedding, dim = 0)

# C3 Test dataset
c3Id = []
c3Label = []
for i, j, l in idPairc3:
    c3Id.append([i, j])
    c3Label.append(int(l))
c3Id = np.array(c3Id)
c3Label = np.array(c3Label)

c3Emdedding = []            
for i, j in c3Id:
    c3Emdedding.append(torch.stack([esm_embed[i], esm_embed[j]], dim = 0))
c3Emdedding = torch.stack(c3Emdedding, dim = 0)

# Training
EPOCH = 40
LR = 0.001
model = MLP()
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_func = nn.BCELoss()
loss_func.cuda()

print('Training Starts')
start2 = time.time()
for epoch in range(1,EPOCH+1):
    start = time.time()
    for step, (batch_x, batch_y) in enumerate(trainLoader):        
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()        
        output = model(batch_x)      
        loss = loss_func(torch.flatten(output).float(), batch_y.cuda().float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end = time.time()
        tmp = len(trainLoader)/width
        if step % tmp < 1:
            print(f'\rEpoch:{epoch} [{"-"*(int(step/tmp) + 1)}->{" "*(width - 1 -int(step/tmp))}][{(end-start):.1f}s][{(end-start2):.0f}s]  ', end='')

# Evaluation       
with torch.no_grad():
    model.to('cpu')
    c2LabelPred = model(c2Emdedding).tolist()
    print(average_precision_score(c2Label, c2LabelPred), roc_auc_score(c2Label, c2LabelPred))
    
    c3LabelPred = model(c3Emdedding).tolist()
    print(average_precision_score(c3Label, c3LabelPred), roc_auc_score(c3Label, c3LabelPred))




