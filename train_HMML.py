import os
import json
import time
import torch
from torch import torch_version
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


def train(model1,model2,train_data,optimizer1,optimizer2,criterion,vocab1,vocab2,batch_size,device,k):
    model1.train()
    model2.train()
    train_size = len(train_data)
    epoch_start = time.time()
    epoch_loss1 = []
    epoch_loss2 = []
    epoch_loss3 = []
    multi_classfier1 = 0
    double_classfier1 = 0
    multi_classfier2 = 0
    double_classfier2 = 0
    multi_classfier3 = 0
    double_classfier3 = 0

    for batch in tqdm(range(0,train_size,batch_size)):
        if batch+batch_size < train_size:
            train_batch = train_data[batch:batch+batch_size]
        else:
            train_batch = train_data[batch:]
        # Model1
        train_lstm = [data['features_content']['code_tokens'] for data in train_batch]
        for features in train_lstm:
            for id in range(len(features)):
                if features[id] not in vocab1.keys():
                    features[id] = 0
                else:
                    features[id] = vocab1[features[id]]
        label = [data['labels_index'] for data in train_batch]
        train_pad = pad_sequence([torch.LongTensor(i) for i in train_lstm],batch_first=True)
        input_batch = Variable(torch.LongTensor(train_pad)).to(device)
        target_batch = torch.tensor(Variable(torch.LongTensor(label)),dtype=torch.float32).to(device)

        # Model2
        train_dataset = []
        for data in (train_batch):
            x = data['features_content']["x"]
            edge_index = data['features_content']["edge_index"]
            edge_index = torch.tensor(edge_index,dtype=torch.long)
            y = data['labels_index']
            y = torch.tensor(y,dtype=torch.float32)
            for i in range(len(x)):
                x[i][0] = vocab2[x[i][0]]
            x = torch.tensor(x,dtype=torch.long)
            train_dataset.append(Data(x=x,edge_index=edge_index,y=y))
        input_data2 = DataLoader(train_dataset,batch_size=batch_size, shuffle=False)
        for i in input_data2:
            data = i
        data.to(device)
        
        # strat train
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        out1, _ = model1(input_batch)
        out2 = model2(data.x,data.edge_index,data.batch)

        output2 = out1
        predict = output2.detach().to('cpu').numpy().tolist()
   \
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

        loss1 = criterion(out1,target_batch)
        loss2 = criterion(out2,target_batch)
        epoch_loss1.append(loss1.item())
        epoch_loss2.append(loss2.item())

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

        out_final = out_final.detach().to('cpu').numpy().tolist()
        for i in out_final:
            for j in range(len(i)):
                if i[j] < 0:
                    i[j] = 0
        out_final = torch.tensor(out_final,device=device)

        loss_final = criterion(out_final,target_batch)
        epoch_loss3.append(loss_final.item())

        loss1.backward()
        loss2.backward()

        optimizer1.step()
        optimizer2.step()
    
    epoch_end = time.time()

    epoch_time = epoch_end-epoch_start
    epoch_lstm_loss = sum(epoch_loss1)/train_size
    epoch_gcn_loss = sum(epoch_loss2)/train_size
    total_loss = sum(epoch_loss3)/train_size

    epoch_lstm_double_acc = double_classfier1/train_size
    epoch_gcn_double_acc = double_classfier2/train_size

    epoch_lstm_multi_acc = multi_classfier1/train_size
    epoch_gcn_multi_acc = multi_classfier2/train_size

    total_double_acc = double_classfier3/train_size
    total_multi_acc = multi_classfier3/train_size

    return epoch_lstm_loss,epoch_gcn_loss,\
        epoch_lstm_double_acc,epoch_gcn_double_acc,\
        epoch_lstm_multi_acc,epoch_gcn_multi_acc,\
        epoch_time,total_loss,total_double_acc,total_multi_acc

    
def validate(model1,model2,dev_data,criterion,vocab1,vocab2,batch_size,device,k):
    model1.eval()
    model2.eval()
    dev_size = len(dev_data)
    epoch_start = time.time()
    epoch_loss1 = []
    epoch_loss2 = []
    epoch_loss3 = []
    multi_classfier1 = 0
    double_classfier1 = 0
    multi_classfier2 = 0
    double_classfier2 = 0
    multi_classfier3 = 0
    double_classfier3 = 0

    for batch in tqdm(range(0,dev_size,batch_size)):
        if batch+batch_size < dev_size:
            dev_batch = dev_data[batch:batch+batch_size]
        else:
            dev_batch = dev_data[batch:]
        # Model1
        dev_lstm = [data['features_content']['code_tokens'] for data in dev_batch]
        for features in dev_lstm:
            for id in range(len(features)):
                if features[id] not in vocab1.keys():
                    features[id] = 0
                else:
                    features[id] = vocab1[features[id]]
        label = [data['labels_index'] for data in dev_batch]
        dev_pad = pad_sequence([torch.LongTensor(i) for i in dev_lstm],batch_first=True)
        input_batch = Variable(torch.LongTensor(dev_pad)).to(device)
        target_batch = torch.tensor(Variable(torch.LongTensor(label)),dtype=torch.float32).to(device)

        # Model2
        dev_dataset = []
        for data in (dev_batch):
            x = data['features_content']["x"]
            edge_index = data['features_content']["edge_index"]
            edge_index = torch.tensor(edge_index,dtype=torch.long)
            y = data['labels_index']
            y = torch.tensor(y,dtype=torch.float32)
            for i in range(len(x)):
                if x[i][0] not in vocab2.keys():
                    x[i][0] = 0
                else:
                    x[i][0] = vocab2[x[i][0]]
            x = torch.tensor(x,dtype=torch.long)
            dev_dataset.append(Data(x=x,edge_index=edge_index,y=y))
        input_data2 = DataLoader(dev_dataset,batch_size=batch_size, shuffle=False)
        for i in input_data2:
            data = i
        data.to(device)
        
        # strat validate
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

            loss1 = criterion(out1,target_batch)
            loss2 = criterion(out2,target_batch)
            epoch_loss1.append(loss1.item())
            epoch_loss2.append(loss2.item())

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

            loss_final = criterion(out_final,target_batch)
            epoch_loss3.append(loss_final.item())
    
    epoch_end = time.time()

    epoch_time = epoch_end-epoch_start
    epoch_lstm_loss = sum(epoch_loss1)/dev_size
    epoch_gcn_loss = sum(epoch_loss2)/dev_size
    total_loss = sum(epoch_loss3)/dev_size

    epoch_lstm_double_acc = double_classfier1/dev_size
    epoch_gcn_double_acc = double_classfier2/dev_size

    epoch_lstm_multi_acc = multi_classfier1/dev_size
    epoch_gcn_multi_acc = multi_classfier2/dev_size

    total_double_acc = double_classfier3/dev_size
    total_multi_acc = multi_classfier3/dev_size

    return epoch_lstm_loss,epoch_gcn_loss,\
        epoch_lstm_double_acc,epoch_gcn_double_acc,\
        epoch_lstm_multi_acc,epoch_gcn_multi_acc,\
        epoch_time,total_loss,total_double_acc,total_multi_acc


def main():
    epoch = 15
    batch_size = 16
    n_hidden = 300
    embedding_dim = 200
    num_classes = 11
    lr = 0.001
    device = 'cuda:1'
    best_score = 0.0
    k = torch.tensor([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5],device=device)
    # k = 0.5

    print("----------Pepare for the Vocabulary Dictionary!----------")
    with open("codegraph_tokens_vocab_dict.json",'r') as f:
        f_tokens = json.load(f)
    vocab_token_size = len(f_tokens)
    with open("codegraph_graph_vocab_dict.json",'r') as f:
        f_graph = json.load(f)
    vocab_graph_size = len(f_graph)

    print("----------Done!----------")
    print("----------Prepare for the Fusion Model!----------")
    model1 = Bilstm_att(vocab_token_size,embedding_dim,n_hidden,num_classes).to(device)
    model2 = GCN(vocab_graph_size,embedding_dim,n_hidden,num_classes).to(device)
    criterion = nn.BCELoss()
    optimizer1 = optim.Adam(model1.parameters(),lr=lr)
    optimizer2 = optim.Adam(model2.parameters(),lr=lr)
    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1,
                                                           mode="max",
                                                           factor=0.5,
                                                           patience=0)
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2,
                                                           mode="max",
                                                           factor=0.5,
                                                           patience=0)
    
    print("----------Done!----------")
    print("----------Let's start our training!----------")
    for epoch_num in range(epoch):
        print("* Training epoch {}:".format(epoch_num))
        with open("train_data100_codegraph.json",'r') as f:
            train_data = json.load(f)
        epoch_lstm_loss,epoch_gcn_loss,\
        epoch_lstm_double_acc,epoch_gcn_double_acc,\
        epoch_lstm_multi_acc,epoch_gcn_multi_acc,\
        total_time,total_loss,total_double_acc,total_multi_acc = train(model1,model2,train_data,optimizer1,optimizer2,
                                                        criterion,f_tokens,f_graph,batch_size,device,k)
                                                                                    
        print("-> Training Model1 loss = {:.4f}, bi_accuracy: {:.4f}%, multi_accuracy:{:.4f}%"
              .format(epoch_lstm_loss, (epoch_lstm_double_acc*100),(epoch_lstm_multi_acc*100)))
        
        print("-> Training Model2 loss = {:.4f}, bi_accuracy: {:.4f}%, multi_accuracy:{:.4f}%"
              .format(epoch_gcn_loss, (epoch_gcn_double_acc*100),(epoch_gcn_multi_acc*100)))
        
        print("-> Training Total time = {:.4f}s, loss = {:.4f}, bi_accuracy: {:.4f}%, multi_accuracy:{:.4f}%"
              .format(total_time,total_loss, (total_double_acc*100),(total_multi_acc*100)))

        k = epoch_lstm_multi_acc/(epoch_lstm_multi_acc+epoch_gcn_multi_acc)

        print("k:",k)
        print("* Validating epoch {}:".format(epoch_num))
        with open("dev_data100_codegraph.json",'r') as f:
            dev_data = json.load(f)
        epoch_lstm_loss,epoch_gcn_loss,\
        epoch_lstm_double_acc,epoch_gcn_double_acc,\
        epoch_lstm_multi_acc,epoch_gcn_multi_acc,\
        total_time,total_loss,total_double_acc,total_multi_acc = validate(model1,model2,dev_data,
                                                        criterion,f_tokens,f_graph,batch_size,device,k)
                                                                                    
        print("-> Validating Model1 loss = {:.4f}, bi_accuracy: {:.4f}%, multi_accuracy:{:.4f}%"
              .format(epoch_lstm_loss, (epoch_lstm_double_acc*100),(epoch_lstm_multi_acc*100)))
        
        print("-> Validating Model2 loss = {:.4f}, bi_accuracy: {:.4f}%, multi_accuracy:{:.4f}%"
              .format(epoch_gcn_loss, (epoch_gcn_double_acc*100),(epoch_gcn_multi_acc*100)))
        
        print("-> Validating Total time = {:.4f}s, loss = {:.4f}, bi_accuracy: {:.4f}%, multi_accuracy:{:.4f}%"
              .format(total_time,total_loss, (total_double_acc*100),(total_multi_acc*100)))
        

        print("")
        
        scheduler1.step(total_multi_acc)
        scheduler2.step(total_multi_acc)

        if total_multi_acc > best_score:
            torch.save({
                'epoch':epoch_num,
                'net1': model1.state_dict(),
                'net2':model2.state_dict(),
                'k':k
            },'HMML_best.pth')

            best_score = total_multi_acc

main()