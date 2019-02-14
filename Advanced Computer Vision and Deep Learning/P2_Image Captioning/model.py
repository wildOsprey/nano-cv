import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        #Embedding layer which encodes words into a vector of embed_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        #LSTM which takes embeddings as an input and return hidden states
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        #Output layer 
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        #Create embeddings for each word in captions. In captions we discard <end> token to avoid a situation if this token comes at the begining 
        #(batch size, embeded size)
        cap_embedding = self.embed(captions[:,:-1])
        
        #Cocatenates feature vector and encoded words vector into the one embeddings vector
        #(batch_size, caption length, embed_size)
        embeddings = torch.cat((features.unsqueeze(1), cap_embedding), 1)
        
        #Passing through lstm and get output and hidden
        #(batch_size, caption length, embed_size) -> (batch_size, caption length, hidden_size)
        lstm_out, self.hidden = self.lstm(embeddings)
        
        #Fully connected layer
        #(batch_size, caption length, hidden_size) -> (batch_size, caption length, vocab_size)
        outputs = self.linear(lstm_out)
        
        return outputs

    
    def sample(self, inputs, hidden=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        res = []
        for i in range(max_len):
            outputs, hidden = self.lstm(inputs, hidden)
            outputs = self.linear(outputs.squeeze(1))
            target_index = outputs.max(1)[1]
            res.append(target_index.item())
            inputs = self.embed(target_index).unsqueeze(1)
        return res