# import os
# import time
import load_data
# import torch
# import torch.nn.functional as F
# from torch.autograd import Variable
# import torch.optim as optim
# import numpy as np
#############################    
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchtext import data
from torchtext import datasets
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score,f1_score,classification_report as cr
from sklearn.metrics import confusion_matrix as cm

import spacy
nlp = spacy.load('en')
import random


import re
#########################
import re
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
URL_REGEX = "(?:http(s)?:\/\/)?[\w.-]+(?:\.[\w\.-]+)+[\w\-\._~:/?#[\]@!\$&'\(\)\*\+,;=.]+"
EMAIL_REGEX = "[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*"
USER_REGEX = "\\@\\w+"

INVISIBLE_REGEX = '[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]'
QUOTATION_REGEX = "[”“❝„\"]+"
APOSTROPHE_REGEX = "[‘´’̇]+"

PRICE_REGEX = '([\$£€¥][0-9]+|[0-9]+[\$£€¥])'
DATE_REGEX = '(?:(?:(?:(?:(?<!:)\\b\\\'?\\d{1,4},? ?)?\\b(?:[Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|May|[Jj]un(?:e)?|[Jj]ul(?:y)?|[Aa]ug(?:ust)?|[Ss]ept?(?:ember)?|[Oo]ct(' \
             '?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)\\b(?:(?:,? ?\\\'?)?\\d{1,4}(?:st|nd|rd|n?th)?\\b(?:[,\\/]? ?\\\'?\\d{2,4}[a-zA-Z]*)?(?: ?- ?\\d{2,4}[a-zA-Z]*)?(?!:\\d{1,' \
             '4})\\b))|(?:(?:(?<!:)\\b\\\'?\\d{1,4},? ?)\\b(?:[Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|May|[Jj]un(?:e)?|[Jj]ul(?:y)?|[Aa]ug(?:ust)?|[Ss]ept?(?:ember)?|[Oo]ct(' \
             '?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)\\b(?:(?:,? ?\\\'?)?\\d{1,4}(?:st|nd|rd|n?th)?\\b(?:[,\\/]? ?\\\'?\\d{2,4}[a-zA-Z]*)?(?: ?- ?\\d{2,4}[a-zA-Z]*)?(?!:\\d{1,4})\\b)?))|(?:\\b(' \
             '?<!\\d\\.)(?:(?:(?:[0123]?[0-9][\\.\\-\\/])?[0123]?[0-9][\\.\\-\\/][12][0-9]{3})|(?:[0123]?[0-9][\\.\\-\\/][0123]?[0-9][\\.\\-\\/][12]?[0-9]{2,3}))(?!\\.\\d)\\b))'
TIME_REGEX = '(?:(?:\d+)?\.?\d+(?:AM|PM|am|pm|a\.m\.|p\.m\.))|(?:(?:[0-2]?[0-9]|[2][0-3]):(?:[0-5][0-9])(?::(?:[0-5][0-9]))?(?: ?(?:AM|PM|am|pm|a\.m\.|p\.m\.))?)'

import spacy
nlp = spacy.load('en')
def tokenize_en(text):
  
  
  text = re.sub('[\u0370-\u03ff]', '', text)  # Greek and Coptic
  text = re.sub('[\u0400-\u052f]', '', text)  # Cyrillic and Cyrillic Supplementary
  text = re.sub('[\u2500-\u257f]', '', text)  # Box Drawing
  text = re.sub('[\u2e80-\u4dff]', '', text)  # from CJK Radicals Supplement
  text = re.sub('[\u4e00-\u9fff]', '', text)  # CJK Unified Ideographs
  text = re.sub('[\ue000-\uf8ff]', '', text)  # Private Use Area
  text = re.sub('[\uff00-\uffef]', '', text)  # Halfwidth and Fullwidth Forms
  text = re.sub('[\ufe30-\ufe4f]', '', text)  # CJK Compatibility Forms

  text = re.sub(INVISIBLE_REGEX, '', text)
  text = re.sub(QUOTATION_REGEX, '\"', text)
  text = re.sub(APOSTROPHE_REGEX, '\'', text)
  text = re.sub(r"\s+", " ", text)
  
  text = re.sub(PRICE_REGEX, r" <PRICE> ", text)
  text = re.sub(TIME_REGEX, r" <TIME> ", text)
  text = re.sub(DATE_REGEX, r" <DATE> ", text)

  text = re.sub(r"([a-zA-Z]+)([0-9]+)", r"\1 \2", text)
  text = re.sub(r"([0-9]+)([a-zA-Z]+)", r"\1 \2", text)
  text = re.sub(r" [0-9]+ ", r" <NUMBER> ", text)

  text = re.sub(r"(\b)([Ii]) 'm", r"\1\2 am", text)
  text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou) 're", r"\1\2 are", text)
  text = re.sub(r"(\b)([Ll]et) 's", r"\1\2 us", text)
  text = re.sub(r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou) 'll", r"\1\2 will", text)
  text = re.sub(r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou) 've", r"\1\2 have", text)
  
  text = re.sub(r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]as|[Ww]ould) n't", r"\1\2 not", text)
  text = re.sub(r"(\b)([Cc]a) n't", r"\1\2n not", text)
  text = re.sub(r"(\b)([Ww]) on't", r"\1\2ill not", text)
  text = re.sub(r"(\b)([Ss])han't", r"\1\2hall not", text)
  text = re.sub(r" n't ", r" not ", text)

  
  text = re.sub(r"([‼.,;:?!…])+", r" \1 ", text)
  text = re.sub(r"([()])+", r" \1 ", text)
  text = re.sub(r"[-]+", r" - ", text)
  text = re.sub(r"[_]+", r" _ ", text)
  text = re.sub(r"[=]+", r" = ", text)
  text = re.sub(r"[\&]+", r" \& ", text)
  text = re.sub(r"[\+]+", r" \+ ", text)

  text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
  text = re.sub(r"what's", "what is", text)
  text = re.sub(r"\'s", "", text)
  text = re.sub(r"\'ve", "have", text)
  text = re.sub(r"can't", "cannot", text)
  text = re.sub(r"n't", "not", text)
  text = re.sub(r"i'm", "i am", text)
  text = re.sub(r"\'re", "are", text)
  text = re.sub(r"\'d", "would", text)
  text = re.sub(r"\'ll", "will", text)
  text = re.sub(r",", "", text)
  text = re.sub(r"\.", "", text)
  text = re.sub(r"!", "!", text)
  text = re.sub(r"\/", "", text)
  text = re.sub(r"\^", "^", text)
  text = re.sub(r"\+", "+", text)
  text = re.sub(r"\-", "-", text)
  text = re.sub(r"\=", "=", text)
  text = re.sub(r"'", "", text)
  text = re.sub(r"<", "", text)
  text = re.sub(r">", "", text)
  text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
  text = re.sub(r":", ":", text)
  text = re.sub(r" e g ", "eg", text)
  text = re.sub(r" b g ", "bg", text)
  text = re.sub(r" u s ", "american", text)
  text = re.sub(r"\0s", "0", text)
  text = re.sub(r"e - mail", "email", text)
  text = re.sub(r"j k", "jk", text)
  text = re.sub(' +', ' ',text)
  tokenized=[tok.text for tok in nlp(text)]
  t=[]
  for tok in tokenized:
    if tok == " " or tok == "  " or tok == "-":
      continue
    else:
      t.append(tok)
#   if len(tokenized) < 3:
#         tokenized += ['<pad>'] * (3 - len(tokenized))
  return t


#########################
from torch.backends import cudnn
seed = 1234

# seed=random.randint(0, 10000)
print(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# seed = 0
# torch.manual_seed(seed)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = False 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
######################################
from models.LSTM import LSTMClassifier
from models.LSTM_Attn import AttentionModel
from models.RCNN import RCNN
from models.CNN import CNN
from models.selfAttention import SelfAttention
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset()

# def clip_gradient(model, clip_value):
#     params = list(filter(lambda p: p.grad is not None, model.parameters()))
#     for p in params:
#         p.grad.data.clamp_(-clip_value, clip_value)
    
# def train_model(model, train_iter, epoch):
#     total_epoch_loss = 0
#     total_epoch_acc = 0
#     model.cuda()
#     optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
#     steps = 0
#     model.train()
# #     for idx, batch in enumerate(train_iter):
#     for batch in train_iter:
#         text = batch.text[0]
#         target = batch.label
#         target = torch.autograd.Variable(target).long()
#         if torch.cuda.is_available():
#             text = text.cuda()
#             target = target.cuda()
#         if (text.size()[0] is not 32):# One of the batch returned by BucketIterator has length different than 32.
#             continue
#         optim.zero_grad()
#         prediction = model(text)
#         loss = loss_fn(prediction, target)
#         num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
#         acc = 100.0 * num_corrects/len(batch)
#         loss.backward()
#         clip_gradient(model, 1e-1)
#         optim.step()
#         steps += 1
        
#         if steps % 100 == 0:
#             print (f'Epoch: {epoch+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')
        
#         total_epoch_loss += loss.item()
#         total_epoch_acc += acc.item()
        
#     return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)
########################################

# learning_rate = 2e-5
batch_size = 64
output_size = 1
hidden_size = 256
N_LAYERS = 2
#changed from 0.2 to 0.4
DROPOUT = 0.2
embedding_length = 100
in_channels=1
out_channels=192
kernel_heights=[2,3,4,5]
stride=1
padding=0
keep_probab=0.4


model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings,N_LAYERS,DROPOUT)
# model = CNN(batch_size, output_size, in_channels, out_channels, kernel_heights, stride, padding, keep_probab, vocab_size, embedding_length, word_embeddings)
# loss_fn = F.cross_entropy

import torch.optim as optim

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
# criterion=F.cross_entropy
model = model.to(device)
criterion = criterion.to(device)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    
    #round predictions to the closest integer
    
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    #print(len((y.data).cpu().numpy()))
    f1=f1_score((y.data).cpu().numpy(),(rounded_preds.data).cpu().numpy(),average='binary')
    y_mini=(y.data).cpu().numpy()
    pred_mini=(rounded_preds.data).cpu().numpy()
#     print(y_mini.shape)
    acc = correct.sum() / len(correct)
    return acc,f1,y_mini,pred_mini
                  
def train(model, iterator, optimizer, criterion):

  epoch_loss = 0
  epoch_acc = 0
  epoch_f1=0
 
  model.train()

  for batch in iterator:
      text= batch.text[0]
      
      target=batch.label
#       target = torch.autograd.Variable(target).long()
      target=target.reshape([target.shape[0],1])
      optimizer.zero_grad()
#       print(batch)
      predictions = model(text)
#       print(predictions.size())
#       predictions=predictions.reshape([predictions.shape[0]])
      loss = criterion(predictions, target)

      acc,f1,y_mini,pred_mini= binary_accuracy(predictions, target)
      #print(type(f1))
      loss.backward()

      optimizer.step()

      epoch_loss += loss.item()
      epoch_acc += acc.item()
      epoch_f1=epoch_f1+f1
  return epoch_loss / len(iterator), epoch_acc / len(iterator),epoch_f1/len(iterator)
########################################
# def eval_model(model, val_iter):
#     total_epoch_loss = 0
#     total_epoch_acc = 0
#     model.eval()
#     with torch.no_grad():
# #         for idx, batch in enumerate(val_iter):
#           for batch in val_iter:
#             text = batch.text[0]
#             if (text.size()[0] is not 32):
#                 continue
#             target = batch.label
#             target = torch.autograd.Variable(target).long()
#             if torch.cuda.is_available():
#                 text = text.cuda()
#                 target = target.cuda()
#             prediction = model(text)
#             loss = loss_fn(prediction, target)
#             num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
#             acc = 100.0 * num_corrects/len(batch)
#             total_epoch_loss += loss.item()
#             total_epoch_acc += acc.item()

#     return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)
	

# learning_rate = 2e-5
# batch_size = 32
# output_size = 2
# hidden_size = 256
# embedding_length = 100

# model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
# loss_fn = F.cross_entropy

# for epoch in range(10):
#     train_loss, train_acc = train_model(model, train_iter, epoch)
#     val_loss, val_acc = eval_model(model, valid_iter)
    
#     print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')
    
# test_loss, test_acc = eval_model(model, test_iter)
# print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')

# ''' Let us now predict the sentiment on a single sentence just for the testing purpose. '''
# test_sen1 = "This is one of the best creation of Nolan. I can say, it's his magnum opus. Loved the soundtrack and especially those creative dialogues."
# test_sen2 = "Ohh, such a ridiculous movie. Not gonna recommend it to anyone. Complete waste of time and money."

# test_sen1 = TEXT.preprocess(test_sen1)
# test_sen1 = [[TEXT.vocab.stoi[x] for x in test_sen1]]

# test_sen2 = TEXT.preprocess(test_sen2)
# test_sen2 = [[TEXT.vocab.stoi[x] for x in test_sen2]]

# test_sen = np.asarray(test_sen1)
# test_sen = torch.LongTensor(test_sen)
# test_tensor = Variable(test_sen, volatile=True)
# test_tensor = test_tensor.cuda()
# model.eval()
# output = model(test_tensor, 1)
# out = F.softmax(output, 1)
# if (torch.argmax(out[0]) == 1):
#     print ("Sentiment: Positive")
# else:
#     print ("Sentiment: Negative")
########################################
def evaluate(model, iterator, criterion):

  epoch_loss = 0
  epoch_acc = 0
  epoch_f1=0
  y_tot=np.array([])
  pred_tot=np.array([])
  model.eval()

  with torch.no_grad():

      for batch in iterator:
          text= batch.text[0]
	  
          predictions = model(text)
# 	  predictions=predictions.reshape([predictions.shape[0]])
          target=batch.label
	  
#         target = torch.autograd.Variable(target).long()
          target=target.reshape([target.shape[0],1])
          loss = criterion(predictions, target)
          
          acc,f1,y_mini,pred_mini = binary_accuracy(predictions, target)

          epoch_loss += loss.item()
          epoch_acc += acc.item()
          epoch_f1+=f1
          y_tot=np.concatenate([y_tot,y_mini.flatten()])
          pred_tot=np.concatenate([pred_tot,pred_mini.flatten()])
  f1=f1_score(y_tot,pred_tot,average='binary')
  f1_macro=f1_score(y_tot,pred_tot,average='macro')
  precision=precision_score(y_tot,pred_tot,average='binary')	
  print(len(y_tot))
  print(cr(y_tot,pred_tot))
  print(cm(y_tot,pred_tot))
  return epoch_loss / len(iterator), epoch_acc / len(iterator),epoch_f1/len(iterator),f1,f1_macro,precision
  
import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
    
    
N_EPOCHS = 20
best_valid_f1 = float(0)
c=0
for epoch in range(N_EPOCHS):

  start_time = time.time()

  train_loss, train_acc,train_f1 = train(model, train_iter, optimizer, criterion)
  valid_loss, valid_acc,valid_f1,f1,f1_macro,valid_precision = evaluate(model, valid_iter, criterion)
  test_loss, test_acc,test_f1,f1_test,f1_macro_test,test_precision = evaluate(model, test_iter, criterion)
  end_time = time.time()

  epoch_mins, epoch_secs = epoch_time(start_time, end_time)

  if f1 > best_valid_f1:
      best_valid_f1 = f1
      c=0
      torch.save(model.state_dict(), 'tut4-model.pt')
  else:
    c=c+1
#   if c==6:
#     print(epoch)
#     break
  print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
  print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Train_f1 : {train_f1:.4f}')
  print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% | Valid_f1 : {f1:.4f}')
  print(f'\t test. Loss: {test_loss:.3f} |  test. Acc: {test_acc*100:.2f}% | test_f1 : {test_f1:.4f}| test_f1_bin : {f1_test:.4f}')



model.load_state_dict(torch.load('tut4-model.pt'))

test_loss, test_acc,test_f1,f1,f1_macro,precision = evaluate(model, test_iter, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%| Test_f1 : {test_f1:.4f}')
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%| Test_f1_bin : {f1:.4f}')
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%| Test_f1_mac : {f1_macro:.4f}')   
###############################################


def predict_sentiment(model):
    model.eval()
    l=[]
    df=pd.read_csv("SubtaskA_Trial_Test_Labeled - Copy.csv")
    with torch.no_grad():
		
	    for i in range(len(df)):
	      tokenized = TEXT.preprocess(df['data'][i])

	      indexed = [TEXT.vocab.stoi[t] for t in tokenized]
	#       print(len(tokenized))
	      test_sen = np.asarray(indexed)
	      test_sen=np.asarray(test_sen)
	      test_sen = torch.LongTensor(test_sen)
	      test_tensor = Variable(test_sen, volatile=True)
	      test_tensor = test_tensor.cuda()
	      test_tensor=test_tensor.reshape([1,test_tensor.shape[0]])
	#       length = [len(indexed)]
	#       tensor = torch.LongTensor(indexed).to(device)

	#       tensor = tensor.unsqueeze(0)
	#       print(test_tensor.size())
	#       length_tensor = torch.LongTensor(length)
	#       test_tensor = Variable(tensor, volatile=True)
	#       test_tensor = test_tensor.cuda()
	#       test_tensor=test_tensor.unsqueeze(1)
	      prediction = torch.sigmoid(model(test_tensor,1))
	#       print(prediction)
	      l.append(((prediction[0][0]).data).cpu().numpy())

    df['preds']=l
    import csv
    df.to_csv('predidctions.csv')
    return(l)
    
    
a=predict_sentiment(model)
