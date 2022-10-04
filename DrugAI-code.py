import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from os import listdir
from os.path import isfile, join

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

import networkx as nx
import random,math,copy,time,timeit


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


import pickle 
from prettytable import PrettyTable
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, log_loss


def setup_seed(seed):
	 torch.manual_seed(seed)
	 torch.cuda.manual_seed_all(seed)
	 np.random.seed(seed)
	 random.seed(seed)
	 torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(0)
 

import dgl
from dgl import model_zoo
from functools import partial



def shuffle_dataset(dataset, seed):
	np.random.seed(seed)
	np.random.shuffle(dataset)
	return dataset


def split_dataset_r(dataset, ratio1, ratio2, ratio3):
	n1 = int(ratio1 * len(dataset))
	n2 = int(ratio2 * len(dataset))
	dataset_1, dataset_2, dataset_3 = dataset[:n1], dataset[n1:(n1+n2)],dataset[(n1+n2):]
	return dataset_1, dataset_2, dataset_3

def split_dataset_n(dataset,n):
	dataset_1, dataset_2 = dataset[:n], dataset[n:]
	return dataset_2, dataset_1


def dump(fm,f):
	import pickle
	file=open(fm,"wb")
	pickle.dump(f,file)
	file.close
	
def load(fm):
	import pickle
	file=open(fm,"rb")
	f=pickle.load(file)
	file.close
	return f



def collate_molgraphs(data):

	
	D,P,compounds, proteins, proteins_masks, d_n2v, p_n2v,f_d,f_p, actions = map(list, zip(*data))

	bg = dgl.batch(compounds)
	bg.set_n_initializer(dgl.init.zero_initializer)
	bg.set_e_initializer(dgl.init.zero_initializer)
	
	actions=torch.stack( [torch.tensor(float(i)) for i in actions])
	proteins = torch.stack([torch.tensor(i).float() for i in proteins], dim=0)
	d_n2v = torch.stack([torch.tensor(i).float()  for i in d_n2v], dim=0)
	p_n2v = torch.stack([torch.tensor(i).float() for i in p_n2v ], dim=0)
	f_d = torch.stack([torch.tensor(i).float() for i in f_d], dim=0)
	f_p = torch.stack([torch.tensor(i).float() for i in f_p], dim=0)
	
	return bg, proteins, d_n2v, p_n2v,f_d,f_p, actions



class CNN(nn.Module):
	def __init__(self, protein_Oridim, feature_size, out_features, max_seq_len, kernels, dropout_rate):
		super(CNN, self).__init__()

		self.dropout_rate= dropout_rate
		self.protein_Oridim=protein_Oridim
		self.feature_size=feature_size
		self.max_seq_len=max_seq_len
		self.kernels = kernels
		self.out_features = out_features

		self.convs = nn.ModuleList([
				nn.Sequential(nn.Conv1d(in_channels=self.protein_Oridim, 
										out_channels=self.feature_size, 
										kernel_size=ks),
							  nn.ReLU(),
							  nn.MaxPool1d(kernel_size=self.max_seq_len-ks+1)
							 )
					 for ks in self.kernels
					])
		self.fc = nn.Linear(in_features=self.feature_size*len(self.kernels),
							out_features=self.out_features)
		self.dropout=nn.Dropout(p=self.dropout_rate)
		
	def forward(self, x):
		

		embedding_x = x.permute(0, 2, 1)
		
		out = [conv(embedding_x) for conv in self.convs] 
		out = torch.cat(out, dim=1)  
		out = out.view(-1, out.size(1)) 
		out = self.dropout(input=out)
		out = self.fc(out)
		return out

	


def GNN(flag,layers,heads):
    if flag=="AttentiveFP":
        
        compound_Encoder1= model_zoo.chem.AttentiveFP(
                                node_feat_size=58,
                                  edge_feat_size=10,
                                  num_layers=layers,
                                  num_timesteps=2,
                                  graph_feat_size=167,
                                  output_size=167,
                                  dropout=0.1)
       
    
    if flag=="GCN":
        
        compound_Encoder1= model_zoo.chem.GCNClassifier(
                            in_feats=58,
                            gcn_hidden_feats=layers,
                            n_tasks=167,
                            classifier_hidden_feats=167,
                            dropout=0.1)
        
        
    if flag=="GAT":
        
        compound_Encoder1= model_zoo.chem.GATClassifier(
                            in_feats=58,gat_hidden_feats=layers, num_heads=heads, n_tasks=167)
      
    
    if flag=="MPNN":
        
        compound_Encoder1= model_zoo.chem.MPNNModel(
                            node_input_dim=58,
                            edge_input_dim=10,
                            output_dim=167,
                            node_hidden_dim=128,
                            edge_hidden_dim=32,
                            num_step_message_passing=layers
                            )
        
    return compound_Encoder1




class Classifier(nn.Module):
	def __init__(self, CNN_fdim, p_n2v_fdim, g_fp_dim, GNN_fdim, d_n2v_fdim, d_fp_dim, CNN, GNN, 
				 dropout_r, DNN_layers, views_flag, GNNs_flag):
		super(Classifier, self).__init__()

		self.views_flag=views_flag
		self.GNNs_flag=GNNs_flag

		self.protein_Encoder=CNN

		self.compound_Encoder= GNN

		self.dropout = nn.Dropout(p=dropout_r)

		self.DNN_layers= DNN_layers
	##protein_feature
		self.CNN_fdim = CNN_fdim
		self.p_n2v_fdim= p_n2v_fdim
		self.g_fp_dim = g_fp_dim
	##compounds_feature	  
		self.GNN_fdim = GNN_fdim
		self.d_n2v_fdim= d_n2v_fdim
		self.d_fp_dim =  d_fp_dim

		self.layer_size = len(self.DNN_layers) + 1
		
		
		
		views_dim_dict={
		"d_GNN+d_N2V+d_fp+g_CNN+p_n2v+g_fp":[self.GNN_fdim + self.d_n2v_fdim + self.d_fp_dim + self.CNN_fdim + self.p_n2v_fdim + self.g_fp_dim] + self.DNN_layers + [1],
		
		"d_N2V+d_fp+g_CNN+p_n2v+g_fp":[self.d_n2v_fdim +self.d_fp_dim + self.CNN_fdim + self.p_n2v_fdim + self.g_fp_dim] + self.DNN_layers + [1],
			
		"d_GNN+d_fp+g_CNN+p_n2v+g_fp":[self.GNN_fdim + self.d_fp_dim + self.CNN_fdim + self.p_n2v_fdim+self.g_fp_dim] + self.DNN_layers + [1],
			
		"d_GNN+d_N2V+g_CNN+p_n2v+g_fp": [self.GNN_fdim + self.d_n2v_fdim + self.CNN_fdim + self.p_n2v_fdim + self.g_fp_dim] + self.DNN_layers + [1],
			
		"d_GNN+d_N2V+d_fp+p_n2v+g_fp": [self.GNN_fdim + self.d_n2v_fdim +self.d_fp_dim + self.p_n2v_fdim +self.g_fp_dim] + self.DNN_layers + [1],
			
		"d_GNN+d_N2V+d_fp+g_CNN+g_fp": [self.GNN_fdim + self.d_n2v_fdim +self.d_fp_dim + self.CNN_fdim + self.g_fp_dim] + self.DNN_layers + [1],
			
		"d_GNN+d_N2V+d_fp+g_CNN+p_n2v": [self.GNN_fdim + self.d_n2v_fdim +self.d_fp_dim + self.CNN_fdim +self.p_n2v_fdim ] + self.DNN_layers + [1],

		"d_GNN+d_fp+g_CNN+g_fp": [self.GNN_fdim+ self.d_fp_dim + self.CNN_fdim+ self.g_fp_dim] + self.DNN_layers + [1],

		"d_GNN+d_N2V+g_CNN+p_n2v": [self.GNN_fdim + self.d_n2v_fdim + self.CNN_fdim + self.p_n2v_fdim ] + self.DNN_layers + [1],

		"d_GNN+g_CNN": [self.GNN_fdim + self.CNN_fdim] + self.DNN_layers + [1]}

		
		self.dims=views_dim_dict[self.views_flag]



		self.predictor = nn.ModuleList([nn.Linear(self.dims[i], self.dims[i+1]) for i in range(self.layer_size)])



	def forward(self,  compounds, proteins, d_n2v, p_n2v, d_FP, g_FP, actions):

		
		

		if self.GNNs_flag=="AttentiveFP":
			compound_feature=self.compound_Encoder(compounds, compounds.ndata['h'], compounds.edata['e'])
		if self.GNNs_flag=="MPNN":
			compound_feature=self.compound_Encoder(compounds, compounds.ndata['h'],compounds.edata['e'])
		if self.GNNs_flag=="GCN" or self.GNNs_flag=="GAT":
			compound_feature=self.compound_Encoder(compounds, compounds.ndata['h'])
		
		compound_feature=compound_feature.squeeze(1)
		protein_feature = self.protein_Encoder(torch.stack([proteins[i] for i in range(len(proteins))]))
		g_fp = torch.stack([g_FP[i] for i in range(len(g_FP))])
		d_fp= torch.stack([d_FP[i] for i in range(len(d_FP))])

		
		
		
		views_feature_dict={
		"d_GNN+d_N2V+d_fp+g_CNN+p_n2v+g_fp":torch.cat((compound_feature, protein_feature, d_fp, g_fp, d_n2v, p_n2v), 1),
		
		"d_N2V+d_fp+g_CNN+p_n2v+g_fp":	  torch.cat((				  protein_feature, d_fp, g_fp, d_n2v, p_n2v), 1),
			
		"d_GNN+d_fp+g_CNN+p_n2v+g_fp":	  torch.cat((compound_feature, protein_feature, d_fp, g_fp,		p_n2v), 1),
			
		"d_GNN+d_N2V+g_CNN+p_n2v+g_fp":	 torch.cat((compound_feature, protein_feature,	   g_fp, d_n2v, p_n2v), 1),
			
		"d_GNN+d_N2V+d_fp+p_n2v+g_fp":	  torch.cat((compound_feature,				  d_fp, g_fp, d_n2v, p_n2v), 1),
			
		"d_GNN+d_N2V+d_fp+g_CNN+g_fp":	  torch.cat((compound_feature, protein_feature, d_fp, g_fp, d_n2v,	  ), 1),
			
		"d_GNN+d_N2V+d_fp+g_CNN+p_n2v":	 torch.cat((compound_feature, protein_feature, d_fp,	   d_n2v, p_n2v), 1),

		"d_GNN+d_fp+g_CNN+g_fp":			torch.cat((compound_feature, protein_feature, d_fp, g_fp,			 ), 1),

		"d_GNN+d_N2V+g_CNN+p_n2v":		  torch.cat((compound_feature, protein_feature,			 d_n2v, p_n2v), 1),

		"d_GNN+g_CNN":					  torch.cat((compound_feature,protein_feature						   ), 1)
		}
		
		

		v_f = views_feature_dict[self.views_flag]

			
		for i, l in enumerate(self.predictor):
			if i==(len(self.predictor)-1):
				v_f = l(v_f)
			else:
				v_f = F.relu(self.dropout(l(v_f)))


		return v_f, actions


def metric(y_label, y_prob):
	y_pred = np.asarray([1 if i else 0 for i in (np.asarray(y_prob) >= 0.5)])
	accuracy=accuracy_score(y_label, y_pred)
	precision=precision_score(y_label, y_pred, average="binary")
	recall=recall_score(y_label, y_pred, average="binary")
	auc=roc_auc_score(y_label, y_prob)
	auprc=average_precision_score(y_label, y_prob)
	f1=f1_score(y_label, y_pred, average="binary")
	loss=log_loss(y_label, y_pred)

	return accuracy, precision, recall, auc, auprc, f1, loss



class DrugAI():
	'''
		DrugAI 
	'''

	def __init__(self,**config):

		self.views_flag = config["views_flag"]

		self.protein_Oridim=config["protein_Oridim"]
		self.feature_size=config["feature_size"]
		self.out_features=config["out_features"]
		self.max_seq_len=config["max_seq_len"]


		self.kernels=config["kernels"]
		self.CNN_fdim=config["CNN_fdim"]
		self.g_fp_dim=config["g_fp_dim"]
		self.p_n2v_fdim=config["p_n2v_fdim"]


		self.dropout_r =config["dropout_r"]


		self.GNNs_flag=config["GNNs_flag"]
		self.layers=config["GNNs_layers"]
		self.heads=config["GNNs_heads"]
		self.GNN_fdim=config["GNN_fdim"]
		self.d_fp_dim=config["d_fp_dim"]
		self.d_n2v_fdim=config["d_n2v_fdim"]
		

		self.DNN_layers=config["DNN_layers"]

		self.batch_size =config["batch_size"]
		self.lr =config["lr"]
		self.decay =config["decay"]
		self.train_epoch=config["train_epoch"]

		self.result_folder = config["result_folder"]


		
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		
		self.protein_Encoder = CNN(self.protein_Oridim, self.feature_size,self.out_features,self.max_seq_len,
								  self.kernels, self.dropout_r) 

		self.compound_Encoder= GNN(self.GNNs_flag,self.layers,self.heads)

		self.model = Classifier(self.CNN_fdim, self.p_n2v_fdim, self.g_fp_dim, self.GNN_fdim,
								self.d_n2v_fdim, self.d_fp_dim,
								self.protein_Encoder, self.compound_Encoder, self.dropout_r, 
								self.DNN_layers, self.views_flag, self.GNNs_flag)





	def vali(self, dataset_vali):
		y_prob = []
		y_label = []

		dataset_loader = DataLoader(dataset=dataset_vali, batch_size=self.batch_size, collate_fn=collate_molgraphs)

		for i, (compounds, proteins, d_n2v, p_n2v, f_d, f_p, actions) in enumerate(dataset_loader):
			score, actions = self.model(compounds.to(self.device), proteins.to(self.device), d_n2v.to(self.device), p_n2v.to(self.device), f_d.to(self.device), f_p.to(self.device), actions.to(self.device))
			Sigm = torch.nn.Sigmoid()
			prob = torch.squeeze(Sigm(score)).detach().cpu().numpy()
			y_label = y_label + actions.detach().cpu().numpy().flatten().tolist()
			y_prob = y_prob + prob.flatten().tolist()

		y_pred = np.asarray([1 if i else 0 for i in (np.asarray(y_prob) >= 0.5)])

		accuracy, precision, recall, auc, auprc, f1, loss=metric(y_label,y_prob)

		return accuracy, precision, recall, auc, auprc, f1, loss, y_pred, y_prob, y_label

	
	
	def train(self, dataset_train, dataset_vali, vali_flag):


		self.model.to(self.device)
		print("if on cuda:",next(self.model.parameters()).is_cuda)

		loss_func = torch.nn.BCEWithLogitsLoss()
		

		optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.decay)

		print('--- Data Preparation ---')


		train_loader = DataLoader(dataset=dataset_train, batch_size=self.batch_size, collate_fn=collate_molgraphs)
		vali_loader = DataLoader(dataset=dataset_vali, batch_size=self.batch_size, collate_fn=collate_molgraphs)

		
		# early stopping
		max_auc = 0

		
		train_metric_header = ["# epoch","Accuracy","Precision","Recall","AUROC", "AUPRC","F1", "log_Loss"] 


		vali_metric_header = ["# epoch","Accuracy","Precision","Recall","AUROC", "AUPRC","F1", "log_Loss"]


		best_metric_header = ["# epoch","Accuracy","Precision","Recall","AUROC", "AUPRC","F1", "log_Loss"] 

		
		table_train = PrettyTable(train_metric_header)	
		table_vali = PrettyTable(vali_metric_header)
		table_vali_best = PrettyTable(best_metric_header)
		
		float2str = lambda x:'%0.6f'%x
		print('--- Go for Training ---')
		# switch to train mode
		
		

		t_start = time.time() 

		for epo in range(self.train_epoch):
			self.model.train()
			y_label_train=[]
			y_prob_train=[]
			for i, (compounds, proteins, d_n2v, p_n2v, f_d, f_p, actions) in enumerate(train_loader):
				score, actions = self.model(compounds.to(self.device), proteins.to(self.device), d_n2v.to(self.device), p_n2v.to(self.device), f_d.to(self.device), f_p.to(self.device), actions.to(self.device))
				prob = torch.squeeze(score, 1)
				loss = loss_func(prob, actions)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				y_label_train= y_label_train + actions.detach().cpu().numpy().flatten().tolist()
				y_prob_train = y_prob_train + prob.detach().cpu().numpy().flatten().tolist()


			metrics_train=metric(y_label_train,y_prob_train)

			print('------ Training ------')
			print('Training Epoch '+ str(epo + 1) + \
				  ' , Accuracy: ' + str(metrics_train[0])[:6] + \
				  ' , Precision: ' + str(metrics_train[1])[:6] + \
				  ' , Recall: ' + str(metrics_train[2])[:6] + \
				  ' , AUROC: ' + str(metrics_train[3])[:6] + \
				  ' , AUPRC: ' + str(metrics_train[4])[:6]+ \
				  ' , F1: '+str(metrics_train[5])[:6] + \
				  ' , Cross-entropy Loss: ' +  str(metrics_train[6])[:6])

			lst_train = ["epoch " + str(epo+1)] + [str(metrics_train[0])[:6], str(metrics_train[1])[:6] ,str(metrics_train[2])[:6], str(metrics_train[3])[:6], str(metrics_train[4])[:6] ,str(metrics_train[5])[:6], str(metrics_train[6])[:6]]

			if vali_flag:
				
				##### validate, select the best model up to now 
				self.model.eval()
				y_label_vali=[]
				y_prob_vali=[]
				for i, (compounds, proteins, d_n2v, p_n2v, f_d, f_p, actions) in enumerate(vali_loader):
					score, actions = self.model(compounds.to(self.device), proteins.to(self.device), d_n2v.to(self.device), p_n2v.to(self.device), f_d.to(self.device), f_p.to(self.device), actions.to(self.device))
					prob = torch.squeeze(score, 1)
					loss = loss_func(prob, actions)
					y_label_vali= y_label_vali + actions.detach().cpu().numpy().flatten().tolist()
					y_prob_vali = y_prob_vali + prob.detach().cpu().numpy().flatten().tolist()

				metrics_vali=metric(y_label_vali,y_prob_vali)

				lst_vali = ["epoch " + str(epo+1)] + [str(metrics_vali[0])[:6], str(metrics_vali[1])[:6] ,str(metrics_vali[2])[:6], str(metrics_vali[3])[:6], str(metrics_vali[4])[:6] ,str(metrics_vali[5])[:6], str(metrics_train[6])[:6]]


				print('------ Validating ------')
				print('Validating Epoch '+ str(epo + 1) + ' , Accuracy: ' + str(metrics_vali[0])[:6] +' , Precision: ' + str(metrics_vali[1])[:6] +' , Recall: ' + str(metrics_vali[2])[:6] +' , AUROC: ' + str(metrics_vali[3])[:6] +' , AUPRC: ' + str(metrics_vali[4])[:6]+' , F1: '+str(metrics_vali[5])[:6] +' , Cross-entropy Loss: ' +  str(metrics_vali[6])[:6])

				table_vali.add_row(lst_vali)

				if metrics_vali[3] > max_auc:
					max_auc_state ={"net":self.model.state_dict(), "optimizer": optimizer.state_dict(),"epoch":epo}
					max_auc = metrics_vali[3] 
					max_auc_index_list=lst_vali
			else:
				if metrics_train[3] > max_auc:
					max_auc_state ={"net":self.model.state_dict(), "optimizer": optimizer.state_dict(),"epoch":epo}
					max_auc = metrics_train[3] 
					max_auc_index_list=lst_train				  

			table_train.add_row(lst_train)

			t_now = time.time()
			print('Training  Epoch ' + str(epo + 1) +' with loss ' + str(loss.detach().cpu().numpy())[:7] +". Total time " + str(int(t_now - t_start)/3600)[:7] + " hours") 

		table_vali_best.add_row(max_auc_index_list)


		if not os.path.exists(self.result_folder):
			os.mkdir(self.result_folder)

		# load early stopped model
		torch.save(max_auc_state, self.result_folder+"/model_"+str(max_auc)+"_AUC.pkl")
		#### after training 
		
		train_file = os.path.join(self.result_folder, "train_markdowntable.txt")
		with open(train_file, 'w') as fp:
			fp.write(table_train.get_string())
			
		test_file = os.path.join(self.result_folder, "Vali_markdowntable.txt")
		with open(test_file, 'w') as fp:
			fp.write(table_vali.get_string())
			
		best_file = os.path.join(self.result_folder, "Vali_best_markdowntable.txt")
		with open(best_file, 'w') as fp:
			fp.write(table_vali_best.get_string())


		print('------ Training Finished ------')
		  

	def predict(self, dataset_predict):


		print('------ predicting ------')
		

		y_prob = []
		y_label = []

		self.model.eval()
		dataset_loader = DataLoader(dataset=dataset_predict, batch_size=self.batch_size, collate_fn=collate_molgraphs)
		for i, (compounds, proteins, d_n2v, p_n2v, f_d, f_p, actions) in enumerate(dataset_loader):
			score, _ = self.model(compounds, proteins, d_n2v, p_n2v, f_d, f_p, actions=None)
			Sigm = torch.nn.Sigmoid()
			prob = torch.squeeze(Sigm(score)).detach().cpu().numpy()
			y_prob = y_prob + prob.flatten().tolist()

		print('------ Have predicted! ------')

		return y_prob


	   
	def load_pretrained(self, path):
		if self.device == 'cuda':
			state_dict = torch.load(path)
		else:
			state_dict = torch.load(path, map_location = torch.device('cpu'))
		self.model.load_state_dict(state_dict["net"])
		print("------ Have loaded! ------")






config={

		"views_flag":"d_GNN+d_N2V+d_fp+g_CNN+p_n2v",

		"protein_Oridim": 20,
		"feature_size":200,
		"out_features":343,
		"max_seq_len":3000,
		"kernels": [3],
		"CNN_fdim": 343,
		"g_fp_dim": 343,
		"p_n2v_fdim":256,

		"dropout_r":  0.1,
	
		"GNNs_flag":"AttentiveFP",
		"GNNs_layers":1,
		"GNNs_heads": None,
		"GNN_fdim": 167,
		"d_fp_dim": 167,
		"d_n2v_fdim":256,
		
		"DNN_layers": [512,128,32,8],

		"batch_size":  256,
		"lr":  1e-3,
		"decay":  1e-4,
		"train_epoch": 60,
		"result_folder":"./Drug_AI"

	   }  
		

(dataset_train,dataset_vali,dataset_test)=load("dataset_train_vali_test_deepWalk")

print('Data load Complished!!!')
for r in range(5):
    for flag in ["d_GNN+g_CNN","d_GNN+d_N2V+g_CNN+p_n2v","d_GNN+d_fp+g_CNN+g_fp","d_N2V+d_fp+g_CNN+p_n2v+g_fp","d_GNN+d_fp+g_CNN+p_n2v+g_fp","d_GNN+d_N2V+g_CNN+p_n2v+g_fp","d_GNN+d_N2V+d_fp+p_n2v+g_fp","d_GNN+d_N2V+d_fp+g_CNN+g_fp","d_GNN+d_N2V+d_fp+g_CNN+p_n2v","d_GNN+d_N2V+d_fp+g_CNN+p_n2v+g_fp"]:
        print(flag+" training ...")
        config["views_flag"]=flag
        config["result_folder"]="./Cuda—Drug_AI_GCN"+"_flag-"+str(flag)+"_Round_"+str(r)
        DrugAI_model=DrugAI(**config)
        DrugAI_model.train(dataset_train,dataset_vali, vali_flag=True)



