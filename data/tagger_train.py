# python3.7 tagger_train.py <train_file_absolute_path> <model_file_absolute_path>

import pickle
import sys
from random import uniform

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
from TorchCRF import CRF
# Copy from Char CNN paper
# abcdefghijklmnopqrstuvwxyz0123456789 -,;.!?:’’’/\|_@#$%ˆ&*˜‘+-=()[]{}
# 70 characters which including 26 English letters, 10 digits, 33 special characters and new line character.
class WordCharCNNEmbedding(nn.Module):
    """Combination between character and word embedding as the
    features for the tagger. The character embedding is built
    upon CNN and pooling layer.
    """
    def __init__(self, word_vocabulary_size):
        super(WordCharCNNEmbedding, self).__init__()
        
        self.char_vocabulary_size=50
        self.char_embedding_dim=5
        self.char_padding_idx=1
        self.L=3
        self.output_size=30
        self.word_padding_idx=1
        self.word_embedding_size=300
        self.word_vocabulary_size=word_vocabulary_size
        self.padding_size=2
        self.pretrained_word_embedding=None
        
        # Init char embeddings
        self.char_embedding = nn.Embedding(self.char_vocabulary_size, self.char_embedding_dim, self.char_padding_idx)
        
        # Convolution for embeddings
        self.conv_embedding = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=1, out_channels=self.output_size, kernel_size=(self.L, self.char_embedding_dim)))
        
        # Word embeddings
        self.word_embedding = nn.Embedding(self.word_vocabulary_size, self.word_embedding_size, self.word_padding_idx)
        

    def forward(self, X, X_word):
        word_size = X.size(1)
        char_embeddings = []
        # We have X in dimension of [batch, words, chars]. To use
        # batch calculation we need to loop over all words and
        # calculate the embedding
        for i in range(word_size):
            # Convert the embedding size from [batch, chars]
            # into [batch, 1, chars]. 1 is our channel for
            # convolution layer later
            x = X[:, i, :].unsqueeze(1)
            # Apply embedding for every characters on batch.
            # The dimension now will be [batch, 1, chars, emb]
            char_embedding = self.char_embedding(x)
            # Apply char embedding with dropout and convolution
            # layers so the dim now will be [batch, conv_size, new_height, 1]
            char_embedding = self.conv_embedding(char_embedding)
            # Remove the last dimension with size 1
            char_embedding = char_embedding.squeeze(-1)
            # Apply pooling layer so the new dim will be [batch, conv_size, 1]
            char_embedding = F.max_pool2d(
                char_embedding,
                kernel_size=(1, char_embedding.size(2)),
                stride=1)
            # Transpose it before we put it into array for later concatenation
            char_embeddings.append(char_embedding.transpose(1, 2))

        # Concatenate the whole char embeddings
        final_char_embedding = torch.cat(char_embeddings, dim=1)
        word_embedding = self.word_embedding(X_word)

        # Combine both character and word embeddings
        result = torch.cat([final_char_embedding, word_embedding], 2)
        return result
    
class BiLstmTagger(nn.Module):
    def __init__(self,
                 embedding,
                 nemb,
                 nhid,
                 nlayers,
                 drop,
                 ntags,
                 batch_first=True):
        
        super(BiLstmTagger, self).__init__()
        
        self.embedding = embedding
        
        self.tagger_rnn = nn.LSTM(
            input_size=nemb,
            hidden_size=nhid,
            num_layers=nlayers,
            dropout=drop,
            bidirectional=True)

        self.projection = nn.Sequential(
            nn.Linear(in_features=nhid * 2, out_features=ntags))

        self.crf_tagger = CRF(ntags)
        self._batch_first = batch_first


    def _rnn_forward(self, x, seq_len):
        packed_sequence = pack_padded_sequence(
            x, seq_len, batch_first=self._batch_first)
        out, _ = self.tagger_rnn(packed_sequence)
        out, lengths = pad_packed_sequence(out, batch_first=self._batch_first)
        projection = self.projection(out)

        return projection

    def forward(self, x, x_word, seq_len, y):
        embed = self.embedding(x, x_word)
        projection = self._rnn_forward(embed, seq_len)
        llikelihood = self.crf_tagger(projection, y)

        return -llikelihood

    def decode(self, x, x_word, seq_len):
        embed = self.embedding(x, x_word)
        projection = self._rnn_forward(embed, seq_len)
        result = self.crf_tagger.decode(projection)

        return result    

def preprocess(train_file):
    file = open(train_file, "r")
    handle = file.read()
    #print(handle)
    sentences = handle.splitlines()
    file.close()
    training_data = []
    tag_to_ix = {}
    word_to_ix = {}
    ix_to_tag = {}
    out = "sentence, char_sentence, tags \n"
    for sentence in sentences:
        sentence = sentence.split(" ")
        words = []
        tags = []
        for idx in range(len(sentence)):
            if(len(sentence[idx].rsplit("/",1)) < 2):
                continue
            word = sentence[idx].rsplit("/",1)[0]
            tag = sentence[idx].rsplit("/",1)[1]
            words.append(word)
            tags.append(tag)
            if(tag not in tag_to_ix):
                tag_to_ix[tag] = len(tag_to_ix)
                ix_to_tag[len(tag_to_ix)] = tag
            if(word not in word_to_ix):
                word_to_ix[word] = len(word_to_ix)
        out += '\"'+str(" ".join(words)) + '\"' +","
        
        out += '\"'+str(" ".join([ str(" ".join(words))[i] + " " for i in range(len(str(" ".join(words))))])) + '\"' +","
        
        out +=   '\"'+ str(" ".join(tags)) + '\"' + "\n"
        
        training_data.append((words, tags))
    out_file = open("dataset.train.csv", "w+")
    out_file.write(out)
    out_file.close()
    return training_data, tag_to_ix, word_to_ix

def get_dataset(base_path,
                filename,
                batch_size,
                pretrained_embedding=None,
                is_inference=False):
    sentence = data.Field(lower=False, include_lengths=True, batch_first=True)
    char_nesting = data.Field(lower=False, tokenize=list)
    char_sentence = data.NestedField(char_nesting, include_lengths=True)
    tags = data.Field(batch_first=True)

    train = data.TabularDataset(
        path = filename + ".train.csv",
        format="csv",
        skip_header=True,
        fields=[(("sentence", "char_sentence"), (sentence, char_sentence)),
                ("tags", tags)])
    tags.build_vocab(train.tags)
    
    sentence.build_vocab(train.sentence, vectors=pretrained_embedding)
    char_sentence.build_vocab(train.char_sentence)

    return sentence, char_sentence, tags

def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
    # use torch library to save model parameters, hyperparameters, etc. to model_file
    training_data, tag_to_ix, word_to_ix = preprocess(train_file)
    
    sentence, char_sentence, tags = get_dataset("","dataset", 1)
    
    embedding = WordCharCNNEmbedding(len(word_to_ix))  
    tagger = BiLstmTagger(
            embedding=embedding,
            nemb= 30 + 300,
            nhid=64,
            nlayers=5,
            drop=0.175,
            ntags=len(tag_to_ix))
    model = nn.DataParallel(tagger)
    tagger_params = filter(lambda p: p.requires_grad, tagger.parameters())
    optimizer = optim.SGD(params=tagger_params, lr=0.1)
    loss_function = nn.NLLLoss()
    
    for epoch in range(10):  
        
        model.zero_grad()
        tag_scores = model(sentence, char_sentence, tags)
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
                
    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
