# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class LSTMClassifier(nn.Module):
	def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights,n_layers,dropout):
		super(LSTMClassifier, self).__init__()
		
		"""
		Arguments
		---------
		batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
		output_size : 2 = (pos, neg)
		hidden_sie : Size of the hidden_state of the LSTM
		vocab_size : Size of the vocabulary containing unique words
		embedding_length : Embeddding dimension of GloVe word embeddings
		weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
		
		"""
		
		self.batch_size = batch_size
		self.output_size = output_size
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length
		self.dropout=nn.Dropout(dropout)
		self.dropout_embd=nn.Dropout(0.5)
		self.relu=nn.ReLU()
		self.word_embeddings = nn.Embedding(vocab_size, embedding_length)# Initializing the look-up table.
		self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False) # Assigning the look-up table to the pre-trained GloVe word embedding.
		self.lstm = nn.LSTM(embedding_length, hidden_size,dropout=dropout)
		
		self.fc1 = nn.Linear(hidden_size, 150)
		nn.init.kaiming_normal_(self.fc1.weight)
		self.fc2 = nn.Linear(150, 25)
		nn.init.kaiming_normal_(self.fc2.weight)
		self.label = nn.Linear(25, output_size)
		nn.init.kaiming_normal_(self.label.weight)
 		
		
	def forward(self, input_sentence, batch_size=None):
	
		""" 
		Parameters
		----------
		input_sentence: input_sentence of shape = (batch_size, num_sequences)
		batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
		
		Returns
		-------
		Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
		final_output.shape = (batch_size, output_size)
		
		"""
		
		''' Here we will map all the indexes present in the input sequence to the corresponding word vector using our pre-trained word_embedddins.'''
		input = self.word_embeddings(input_sentence) # embedded input of shape = (batch_size, num_sequences,  embedding_length)
		input = input.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)
		input=self.dropout_embd(input)
# 		if batch_size is None:
# 			h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda()) # Initial hidden state of the LSTM
# 			c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda()) # Initial cell state of the LSTM
# 		else:
# 			h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
# 			c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
		output, (final_hidden_state, final_cell_state) = self.lstm(input)
# 	        final_hidden_state = self.dropout(self.relu(torch.cat((final_hidden_state[-2,:,:], final_hidden_state[-1,:,:]), dim = 1)))
# 	        final_hidden_state=self.dropout(self.relu(final_hidden_state.squeeze(0)))
#                 final_hidden_state=final_hidden_state[-1]
# 	        final_hidden_state=self.dropout(final_hidden_state)
		final_output = self.relu(self.fc1(final_hidden_state[-1]))
		final_output=self.dropout(final_output)
		final_output = self.relu(self.fc2(final_output))
		final_output=self.dropout(final_output)
		final_output = self.label(final_output) # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)
		
		return final_output
