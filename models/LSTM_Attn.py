# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

class AttentionModel(torch.nn.Module):
	def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights,n_layers,dropout):
		super(AttentionModel, self).__init__()
		
		"""
		Arguments
		---------
		batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
		output_size : 2 = (pos, neg)
		hidden_sie : Size of the hidden_state of the LSTM
		vocab_size : Size of the vocabulary containing unique words
		embedding_length : Embeddding dimension of GloVe word embeddings
		weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
		
		--------
		
		"""
		
		self.batch_size = batch_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length
		
		self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
		self.word_embeddings.weights = nn.Parameter(weights, requires_grad=False)
		self.lstm = nn.LSTM(embedding_length, hidden_size,num_layers=n_layers,bidirectional=True,dropout=dropout)
		self.fc1 = nn.Linear(hidden_size*2, 150)
# 		nn.init.kaiming_normal_(self.fc1.weight)
		self.fc2 = nn.Linear(150, 25)
# 		nn.init.kaiming_normal_(self.fc2.weight)
		self.label = nn.Linear(25, output_size)
# 		nn.init.kaiming_normal_(self.label.weight)
		#self.attn_fc_layer = nn.Linear()
		self.dropout = nn.Dropout(dropout)
		self.dropout_embd = nn.Dropout(0.5)
		self.relu=nn.ReLU()
		
	def attention_net(self, lstm_output, final_state):

		""" 
		Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
		between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.
		
		Arguments
		---------
		
		lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
		final_state : Final time-step hidden state (h_n) of the LSTM
		
		---------
		
		Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
				  new hidden state.
				  
		Tensor Size :
					hidden.size() = (batch_size, hidden_size)
					attn_weights.size() = (batch_size, num_seq)
					soft_attn_weights.size() = (batch_size, num_seq)
					new_hidden_state.size() = (batch_size, hidden_size)
					  
		"""
		
		hidden = final_state.squeeze(0)
		attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
		soft_attn_weights = F.softmax(attn_weights, 1)
		new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
		
		return new_hidden_state
	
	def forward(self, input_sentences, batch_size=None):
	
		""" 
		Parameters
		----------
		input_sentence: input_sentence of shape = (batch_size, num_sequences)
		batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
		
		Returns
		-------
		Output of the linear layer containing logits for pos & neg class which receives its input as the new_hidden_state which is basically the output of the Attention network.
		final_output.shape = (batch_size, output_size)
		
		"""
		
		input = self.word_embeddings(input_sentences)
		input = input.permute(1, 0, 2)
		input=self.dropout_embd(input)
# 		if batch_size is None:
# 			h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
# 			c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
# 		else:
# 			h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
# 			c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
			
		output, (final_hidden_state, final_cell_state) = self.lstm(input) # final_hidden_state.size() = (1, batch_size, hidden_size) 
		output = output.permute(1, 0, 2) # output.size() = (batch_size, num_seq, hidden_size)
		final_hidden_state = self.dropout(torch.cat((final_hidden_state[-2,:,:], final_hidden_state[-1,:,:]), dim = 1))
		attn_output = self.dropout(self.relu(self.attention_net(output, final_hidden_state)))
		logits = self.relu(self.fc1(attn_output))
		logits=self.dropout(logits)
		logits = self.relu(self.fc2(logits))
		logits=self.dropout(logits)
		logits=self.label(logits)
		return logits
