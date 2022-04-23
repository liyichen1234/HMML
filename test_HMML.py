import os
import json
import time
import torch
import torch.nn as nn
from model import Bilstm_att,GCN
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from torch_geometric.data import *
from torch_geometric.loader import DataLoader,DataListLoader
from torch_geometric.nn import DataParallel

class Evaluation:
    def __init__(self,true_positive,false_positive,false_negative):
        self.TP = true_positive
        self.FP = false_positive
        self.FN = false_negative

    
    def getP(self):
        if self.TP == 0:
            return 0
        return self.TP/(self.TP+self.FP)
    
    def getR(self):
        if self.TP == 0:
            return 0
        return self.TP/(self.TP+self.FN)
    
    def getF1(self):
        if self.TP == 0:
            return 0
        return 2*self.getP()*self.getR()/(self.getR()+self.getP())


def test():
    batch_size = 16
    n_hidden = 300
    embedding_dim = 200
    num_classes = 11
    device = 'cuda:1'

    with open("codegraph_tokens_vocab_dict.json",'r') as f:
        f_tokens = json.load(f)
    vocab_token_size = len(f_tokens)
    with open("codegraph_graph_vocab_dict.json",'r') as f:
        f_graph = json.load(f)
    vocab_graph_size = len(f_graph)

    checkpoint = torch.load('HMML_best.pth')
    model1 = Bilstm_att(vocab_token_size,embedding_dim,n_hidden,num_classes).to(device)
    model1.eval()
    model1.load_state_dict(checkpoint['net1'])
    model2 = GCN(vocab_graph_size,embedding_dim,n_hidden,num_classes).to(device)
    model2.eval()
    model2.load_state_dict(checkpoint['net2'])
    k = checkpoint['k']

    with open("test_data100_codegraph.json",'r') as f:
        test_data = json.load(f)

   
    test_size = len(test_data)
    
    multi_classfier1 = 0
    double_classfier1 = 0
    multi_classfier2 = 0
    double_classfier2 = 0
    multi_classfier3 = 0
    double_classfier3 = 0

    true_positive1 = 0
    false_positive1 = 0
    false_negative1 = 0

    true_positive2 = 0
    false_positive2 = 0
    false_negative2 = 0

    true_positive3 = 0
    false_positive3 = 0
    false_negative3 = 0

    evaluation_res = [Evaluation(0,0,0) for _ in range(11)]
    all_eva = evaluation_res[-1]

    for batch in tqdm(range(0,test_size,batch_size)):
        if batch+batch_size < test_size:
            test_batch = test_data[batch:batch+batch_size]
        else:
            test_batch = test_data[batch:]
        # Model1
        dev_lstm = [data['features_content']['code_tokens'] for data in test_batch]
        for features in dev_lstm:
            for id in range(len(features)):
                if features[id] not in f_tokens.keys():
                    features[id] = 0
                else:
                    features[id] = f_tokens[features[id]]
        label = [data['labels_index'] for data in test_batch]
        dev_pad = pad_sequence([torch.LongTensor(i) for i in dev_lstm],batch_first=True)
        input_batch = Variable(torch.LongTensor(dev_pad)).to(device)
        target_batch = torch.tensor(Variable(torch.LongTensor(label)),dtype=torch.float32).to(device)

        # Model2
        dev_dataset = []
        for data in (test_batch):
            x = data['features_content']["x"]
            edge_index = data['features_content']["edge_index"]
            edge_index = torch.tensor(edge_index,dtype=torch.long)
            y = data['labels_index']
            y = torch.tensor(y,dtype=torch.float32)
            for i in range(len(x)):
                if x[i][0] not in f_graph.keys():
                    x[i][0] = 0
                else:
                    x[i][0] = f_graph[x[i][0]]
            x = torch.tensor(x,dtype=torch.long)
            dev_dataset.append(Data(x=x,edge_index=edge_index,y=y))
        input_data2 = DataLoader(dev_dataset,batch_size=batch_size, shuffle=False)
        for i in input_data2:
            data = i
        data.to(device)
        
        # strat test
        with torch.no_grad():
            out1, _ = model1(input_batch)
            out2 = model2(data.x,data.edge_index,data.batch)

            output2 = out1
            predict = output2.detach().to('cpu').numpy().tolist()
            for i in predict:
                for j in range(len(i)):
                    if i[j] >= 0.5:
                        i[j] = 1
                    else:
                        i[j] = 0
            for i in range(len(label)):
                if label[i][0] == predict[i][0]:
                    double_classfier1 += 1
                if label[i] == predict[i]:
                    multi_classfier1 += 1
                if  (predict[i][0] == 1):
                    if label[i] == predict[i]:
                        true_positive1 += 1
                    else:
                        false_positive1 += 1
                if (label[i][0] == 1):
                    if label[i] != predict[i]:
                        false_negative1 += 1 
              
            
            output3 = out2
            predict2 = output3.detach().to('cpu').numpy().tolist()

            for i in predict2:
                for j in range(len(i)):
                    if i[j] >= 0.5:
                        i[j] = 1
                    else:
                        i[j] = 0
        
            for i in range(len(label)):
                if label[i][0] == predict2[i][0]:
                    double_classfier2 += 1
                if label[i] == predict2[i]:
                    multi_classfier2 += 1

                if  (predict2[i][0] == 1):
                    if label[i] == predict2[i]:
                        true_positive2 += 1
                    else:
                        false_positive2 += 1
                if (label[i][0] == 1):
                    if label[i] != predict2[i]:
                        false_negative2 += 1 
           
            out_final = k*out1+(1-k)*out2
            output4 = out_final
            predict3 = output4.detach().to('cpu').numpy().tolist()

            for i in predict3:
                for j in range(len(i)):
                    if i[j] >= 0.5:
                        i[j] = 1
                    else:
                        i[j] = 0
        
            for i in range(len(label)):
                if label[i][0] == predict3[i][0]:
                    double_classfier3 += 1
                if label[i] == predict3[i]:
                    multi_classfier3 += 1

                if  (predict3[i][0] == 1):
                    if label[i] == predict3[i]:
                        true_positive3 += 1
                    else:
                        false_positive3 += 1
                if (label[i][0] == 1):
                    if label[i] != predict3[i]:
                        false_negative3 += 1 

                if  (predict[i][0] == 1 and predict[i].count(1) > 2):
                    if label[i] == predict[i]:
                        all_eva.TP += 1
                    else:
                        all_eva.FP += 1
                if (label[i][0] == 1 and label[i].count(1) > 2):
                    if label[i] != predict[i]:
                        all_eva.FN += 1 
            
                if predict[i][0] == 1:
                    for j in range(1,len(predict[i])):
                        if predict[i][j] == 1:
                            if predict[i][j] == label[i][j]:
                                evaluation_res[j-1].TP += 1
                            else:
                                evaluation_res[j-1].FP += 1

                if label[i][0] == 1:
                    for j in range(1,len(label[i])):
                        if label[i][j] == 1:
                            if predict[i][j] != label[i][j]:
                                evaluation_res[j-1].FN += 1
                



    epoch_lstm_double_acc = double_classfier1/test_size
    epoch_gcn_double_acc = double_classfier2/test_size

    epoch_lstm_multi_acc = multi_classfier1/test_size
    epoch_gcn_multi_acc = multi_classfier2/test_size

    total_double_acc = double_classfier3/test_size
    total_multi_acc = multi_classfier3/test_size

    lstm_P = true_positive1/(true_positive1+false_positive1)
    lstm_R = true_positive1/(true_positive1+false_negative1)
    lstm_F = 2*lstm_P*lstm_R/(lstm_P+lstm_R)

    gcn_P = true_positive2/(true_positive2+false_positive2)
    gcn_R = true_positive2/(true_positive2+false_negative2)
    gcn_F = 2*gcn_P*gcn_R/(gcn_P+gcn_R)

    total_P = true_positive3/(true_positive3+false_positive3)
    total_R = true_positive3/(true_positive3+false_negative3)
    total_F = 2*total_P*total_R/(total_P+total_R)

    for i in evaluation_res:
        print("P:",i.getP(),"R:",i.getR(),"F:",i.getF1())
test()

