# _*_ coding: utf-8 _*_

import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
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

def load_dataset(test_sen=None):

    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
    Field : A class that stores information about the way of preprocessing
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                 will pad each sequence to have a fix length of 200.
                 
    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.
                  
    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.
    
    """
    
#     tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenize_en, lower=True, include_lengths=True, batch_first=True, fix_length=40)
    LABEL = data.LabelField(dtype=torch.float)
    fields = [(None,None),(None,None),('text', TEXT),('label', LABEL)]
    train_data, valid_data, test_data = data.TabularDataset.splits(
                                        path = '',
                                        train = 'V1.4_Training.csv',
                                        validation = 'SubtaskB_Trial_Test_Labeled - Copy.csv',
                                        test = 'SubtaskB_EvaluationData_labeled.csv',
#                                         train = 'train_spacy.csv',
#                                         validation = 'valid_spacy.csv',
#                                         test = 'test_spacy.csv',
#                                         #sort_key=lambda x: len(x.Text),
                                        format = 'csv',
                                        fields = fields,
                                        skip_header = True
) 
    print(vars(train_data[0]))
    TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=100))
    LABEL.build_vocab(train_data)

    word_embeddings = TEXT.vocab.vectors
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))

#     train_data, valid_data = train_data.split() # Further splitting of training_data to create new training_data & validation_data
    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=64, sort_key=lambda x: len(x.text), repeat=False, shuffle=True,device=device)

    '''Alternatively we can also use the default configurations'''
    # train_iter, test_iter = datasets.IMDB.iters(batch_size=32)

    vocab_size = len(TEXT.vocab)

    return TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter
