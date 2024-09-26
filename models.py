import torch
from pytorch_pretrained_bert import BertModel
from torch import nn
from torch.nn import functional as F

from config import HIDDEN_DIM, MLP_DIM

class BertMultiClassifier(nn.Module):
    def __init__(self, bert_model_path, labels_count, hidden_dim = HIDDEN_DIM, dropout=0.1):
        super(BertMultiClassifier, self).__init__()


        self.config = {
            'bert_model_path': bert_model_path,
            'labels_count': labels_count,
            'hidden_dim': hidden_dim,
            'dropout': dropout
        }

        self.bert = BertModel.from_pretrained(bert_model_path)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, labels_count)
        self.sigmoid = nn.Sigmoid()


    def forward(self, tokens, masks):
        _, pooled_output = self.bert(tokens, attention_mask=masks)
        dropout_output = self.dropout(pooled_output)

        linear_output = self.linear(dropout_output)
        proba = self.sigmoid(linear_output)

        return proba







class ExtraBertMultiClassifier(nn.Module):
    def __init__(self, bert_model_path, labels_count, hidden_dim = HIDDEN_DIM, mlp_dim =MLP_DIM, extras_dim = 6 ,dropout=0.1):
        super(ExtraBertMultiClassifier, self).__init__()


        self.config = {
            'bert_model_path': bert_model_path,
            'labels_count': labels_count,
            'hidden_dim': hidden_dim,
            'mlp_dim':mlp_dim,
            'extras_dim': extras_dim,
            'dropout': dropout

        }

        self.bert = BertModel.from_pretrained(bert_model_path)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + extras_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, labels_count)
        )
        


        self.softmax  = nn.Softmax()
    
    def forward(self, tokens, masks, extras):
        '''
            tokens: [batch_size, seq_len, 768]
            masks: [batch_size, seq_len, 1]
            extras: [batch_size, seq_lem, extras_dim]
        '''
        _, pooled_output = self.bert(tokens, attention_mask=masks, output_all_encoded_layers=False)
        dropout_output = self.dropout(pooled_output)

        concat_output = torch.cat([dropout_output, extras], dim=1)

        mlp_output = self.mlp(concat_output)
        proba = self.softmax(mlp_output)

        return proba







class ExtraBertTextRCNNMultiClassifier(nn.Module):
    def __init__(self, bert_model_path, labels_count, rnn_hidden_dim=256, cnn_kernel_size=3,   
                 rnn_type='LSTM', num_layers=1, hidden_dim=HIDDEN_DIM, mlp_dim=MLP_DIM, extras_dim=6, dropout=0.1):  
        super(ExtraBertTextRCNNMultiClassifier, self).__init__()  

        # 存储模型的超参数  
        self.config = {  
            'bert_model_path': bert_model_path,  
            'labels_count': labels_count,  
            'rnn_hidden_dim': rnn_hidden_dim,  
            'cnn_kernel_size': cnn_kernel_size,  
            'rnn_type': rnn_type,  
            'num_layers': num_layers,  
            'hidden_dim': hidden_dim,  
            'mlp_dim': mlp_dim,  
            'extras_dim': extras_dim,  
            'dropout': dropout  
        }  

        self.bert = BertModel.from_pretrained(bert_model_path)  

        # 初始化RNN（可以是LSTM）        
        self.rnn = nn.LSTM(input_size=hidden_dim+extras_dim, hidden_size=rnn_hidden_dim,  
                           num_layers=num_layers, batch_first=True, bidirectional=False)  

        # 初始化CNN层用于特征提取  
        self.cnn = nn.Conv1d(in_channels=rnn_hidden_dim,  # 如果是双向LSTM， 就要 2* rnn_hidden_dim
                             out_channels=rnn_hidden_dim,  
                             kernel_size=cnn_kernel_size,  
                             padding=cnn_kernel_size // 2)  
        # 这种填充方式可以确保经过卷积操作后，输出的序列长度与输入长度相同，避免信息丢失。
        '''
         input_max_len: L
         kernel_size: K
         padding: P = K // 2
         output_max_len: L_out
         stride: 1
         
         L_out = (L + 2 * P - K)/stride + 1
        
        '''

      
        self.dropout = nn.Dropout(dropout)  

        # MLP全连接层  
        self.mlp = nn.Sequential(  
            nn.Linear(rnn_hidden_dim, mlp_dim),  
            nn.ReLU(),  
            nn.Linear(mlp_dim, mlp_dim),  
            nn.ReLU(),  
            nn.Linear(mlp_dim, labels_count)  
        )  

        # Softmax层用于输出概率  
        self.softmax = nn.Softmax(dim=1)  

    def forward(self, tokens, masks, extras):  
        '''  
        tokens: [batch_size, seq_len, hidden_dim]  
        masks: [batch_size, seq_len]  
        extras: [batch_size, seq_len, extras_dim]  
        '''  
        # 通过BERT模型获取最后一个隐藏层的输出和池化输出  
        sequence_output, pooler_output = self.bert(tokens, attention_mask=masks, output_all_encoded_layers=False)  
        
        sequence_output = torch.cat([sequence_output, extras], dim=1)  
        
        # 通过RNN  
        rnn_output, hidden_output = self.rnn(sequence_output)  

        # 转换维度使其适应卷积层 (batch_size, channels, seq_len)  
        rnn_output = rnn_output.permute(0, 2, 1)  

        # 通过卷积层提取特征  
        cnn_output = self.cnn(rnn_output)  

        # 转换卷积输出维度  
        cnn_output = cnn_output.permute(0, 2, 1)   # (batch_size, seq_len, rnn_dim)  

        # 对句子长度这维做pooling， 得到一个用于文本分类的表征  
        text_features = torch.mean(cnn_output, dim=1)  

        # 应用dropout， 随机置0
        dropout_output = self.dropout(text_features)  

        # # 拼接dropout后的输出和额外输入信息  
        # concat_output = torch.cat([dropout_output, extras], dim=1)  

        # 通过MLP处理拼接后的输出  
        mlp_output = self.mlp(dropout_output)  

        # 输出概率  
        proba = self.softmax(mlp_output)  

        return proba  



class ExtraDiyBertTextRCNNMultiClassifier(nn.Module):
    pass


class LinearMultiClassifier(nn.Module):
    def __init__(self, labels_count, extras_dim=6, dropout=0.1):
        super().__init__()

        self.config = {
            'labels_count': labels_count,
            'extras_dim': extras_dim,
        }
        self.linear = nn.Linear(extras_dim, labels_count)
        self.softmax = nn.Softmax()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, extras):
        lin_output = self.linear(extras)
        # proba = self.sigmoid(mlp_output)
        proba = self.softmax(lin_output)

        return proba






class ExtraMultiClassifier(nn.Module):
    def __init__(self, labels_count, mlp_dim=100, extras_dim=6, dropout=0.1):
        super().__init__()

        self.config = {
            'labels_count': labels_count,
            'mlp_dim': mlp_dim,
            'extras_dim': extras_dim,
            'dropout': dropout,
        }

        self.mlp = nn.Sequential(
            nn.Linear(extras_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, labels_count)
        )
        self.softmax = nn.Softmax()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, extras):

        mlp_output = self.mlp(extras)
        # proba = self.sigmoid(mlp_output)
        proba = self.softmax(mlp_output)
        
        return proba