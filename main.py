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
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchtext import data
from torchtext import datasets
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,classification_report as cr
from sklearn.metrics import confusion_matrix as cm

import random
import re
from torch.backends import cudnn
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# seed = 0
# torch.manual_seed(seed)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.enabled = False 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
######################################
from models.LSTM import LSTMClassifier
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
batch_size = 32
output_size = 1
hidden_size = 256
embedding_length = 100

model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
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
          target=batch.label
#       target = torch.autograd.Variable(target).long()
          target=target.reshape([batch.numel(),1])
          loss = criterion(predictions, target)

          acc,f1,y_mini,pred_mini = binary_accuracy(predictions, target)

          epoch_loss += loss.item()
          epoch_acc += acc.item()
          epoch_f1+=f1
          y_tot=np.concatenate([y_tot,y_mini])
          pred_tot=np.concatenate([pred_tot,pred_mini])
  f1=f1_score(y_tot,pred_tot,average='binary')
  f1_macro=f1_score(y_tot,pred_tot,average='macro')
  print(len(y_tot))
  print(cr(y_tot,pred_tot))
  print(cm(y_tot,pred_tot))
  return epoch_loss / len(iterator), epoch_acc / len(iterator),epoch_f1/len(iterator),f1,f1_macro
  
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
  valid_loss, valid_acc,valid_f1,f1,f1_macro = evaluate(model, valid_iter, criterion)

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
  print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%| Train_f1 : {train_f1:.4f}')
  print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%| Valid_f1 : {valid_f1:.4f}')



  model.load_state_dict(torch.load('tut4-model.pt'))

test_loss, test_acc,test_f1,f1,f1_macro = evaluate(model, test_iter, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%| Test_f1 : {test_f1:.4f}')
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%| Test_f1_bin : {f1:.4f}')
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%| Test_f1_mac : {f1_macro:.4f}')   
###############################################
