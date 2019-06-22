import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class CNN_LSTM(nn.Module):
	def __init__(self, batch_size, output_size, hidden_size,n_layers,dropout,in_channels, out_channels, kernel_heights, stride, padding, keep_probab, vocab_size, embedding_length, weights):
		super(CNN, self).__init__()
		
		"""
		Arguments
		---------
		batch_size : Size of each batch which is same as the batch_size of the data returned by the TorchText BucketIterator
		output_size : 1 = (pos, neg)
		in_channels : Number of input channels. Here it is 1 as the input data has dimension = (batch_size, num_seq, embedding_length)
		out_channels : Number of output channels after convolution operation performed on the input matrix
		kernel_heights : A list consisting of 3 different kernel_heights. Convolution will be performed 3 times and finally results from each kernel_height will be concatenated.
		keep_probab : Probability of retaining an activation node during dropout operation
		vocab_size : Size of the vocabulary containing unique words
		embedding_length : Embedding dimension of GloVe word embeddings
		weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table
		--------
		
		"""
		self.batch_size = batch_size
		self.output_size = output_size
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_heights = kernel_heights
		self.stride = stride
		self.padding = padding
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length
		
		self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
		self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)
		self.lstm = nn.LSTM(embedding_length, hidden_size,num_layers=n_layers,bidirectional=True,dropout=dropout)
		self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], embedding_length), stride, padding)
		self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], embedding_length), stride, padding)
		self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], embedding_length), stride, padding)
		self.conv4 = nn.Conv2d(in_channels, out_channels, (kernel_heights[3], embedding_length), stride, padding)
		self.dropout = nn.Dropout(keep_probab)
		self.fc1 = nn.Linear(len(kernel_heights)*out_channels+hidden_size*2,300)
		self.fc2 = nn.Linear(300,150)
		self.fc3 = nn.Linear(150,75)
		self.label = nn.Linear(75,output_size)
		self.dropout_embd = nn.Dropout(0.5)
		
		
	def conv_block(self, input, conv_layer):
		
		conv_out = conv_layer(input)# conv_out.size() = (batch_size, out_channels, dim, 1)
		activation = F.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim1)
		max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)# maxpool_out.size() = (batch_size, out_channels)
		
		return max_out
    
    
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
		The idea of the Convolutional Neural Netwok for Text Classification is very simple. We perform convolution operation on the embedding matrix 
		whose shape for each batch is (num_seq, embedding_length) with kernel of varying height but constant width which is same as the embedding_length.
		We will be using ReLU activation after the convolution operation and then for each kernel height, we will use max_pool operation on each tensor 
		and will filter all the maximum activation for every channel and then we will concatenate the resulting tensors. This output is then fully connected
		to the output layers consisting two units which basically gives us the logits for both positive and negative classes.
		
		Parameters
		----------
		input_sentences: input_sentences of shape = (batch_size, num_sequences)
		batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
		
		Returns
		-------
		Output of the linear layer containing logits for pos & neg class.
		logits.size() = (batch_size, output_size)
		
		"""
		
		input = self.word_embeddings(input_sentences)
		# input.size() = (batch_size, num_seq, embedding_length)
		input_cnn = input.unsqueeze(1)
		input=self.dropout_embd(input)
                input_cnn=self.dropout_embd(input_cnn)
		# input.size() = (batch_size, 1, num_seq, embedding_length)
		max_out1 = self.conv_block(input_cnn, self.conv1)
		max_out2 = self.conv_block(input_cnn, self.conv2)
		max_out3 = self.conv_block(input_cnn, self.conv3)
		max_out4 = self.conv_block(input_cnn, self.conv4)
		
		all_out = torch.cat((max_out1, max_out2, max_out3, max_out4), 1)
		# all_out.size() = (batch_size, num_kernels*out_channels)
                output, (final_hidden_state, final_cell_state) = self.lstm(input) # final_hidden_state.size() = (1, batch_size, hidden_size) 
		output = output.permute(1, 0, 2) # output.size() = (batch_size, num_seq, hidden_size)
		final_hidden_state = self.dropout(torch.cat((final_hidden_state[-2,:,:], final_hidden_state[-1,:,:]), dim = 1))
		attn_output = self.dropout(self.relu(self.attention_net(output, final_hidden_state)))
                all_out = torch.cat((all_out,attn_output), dim = 1)
		fc_in = self.dropout(all_out)
		logits = F.relu(self.fc1(fc_in))
		logits=self.dropout(logits)
		logits = F.relu(self.fc2(logits))
		logits=self.dropout(logits)
		logits = F.relu(self.fc3(logits))
		logits=self.dropout(logits)
		
		# fc_in.size()) = (batch_size, num_kernels*out_channels)
		logits = self.label(logits)
		
		return logits
