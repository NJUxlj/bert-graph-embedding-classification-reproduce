import re

'''
DataLoader: 逐批次加载数据
RandomSampler: 打乱数据样本顺序。
SequentialSampler: 按顺序加载数据样本。

TensorDataset:
    功能：将多个张量(Tensors)封装到一个数据集中。
    应用场景：适用于输入数据和目标标签一起形成一个数据集，方便DataLoader处理。


'''
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
# from keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import pad_sequences

# from keras_preprocessing.sequence import pad_sequences

# from keras._tf_keras.keras.utils import pad_sequences

from tensorflow.keras.preprocessing.sequence import pad_sequences

# 计算各类的精度，召回率和F1分数
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_fscore_support

import matplotlib.pyplot as plt
import re

import numpy as np
import torch



# 返回数据项的同时也返回下标
class TensorIndexDataset(TensorDataset):
    def __getitem__(self, index):
        """
        Returns in addition to the actual data item also its index (useful when assign a prediction to a item)
        """
        return index, super().__getitem__(index)




def text_to_train_tensors(texts, tokenizer, max_seq_length):
    '''
    设置 word embedding 层
    '''
    
    # 切片 [:max_seq_length - 1]：控制序列的最大长度（减去1个位置给"[CLS]"），超出部分被截断。

    train_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:max_seq_length - 1], texts)) # 分词
    train_tokens_ids = list(map(tokenizer.convert_tokens_to_ids, train_tokens)) # 将每个单词转换为对应的索引， 索引对应词嵌入向量
    train_tokens_ids = pad_sequences(train_tokens_ids, maxlen=max_seq_length, truncating="post", padding="post",
                                     dtype="int")

    # 每个 ii 都是一个嵌入向量
    '''
     目的：生成一个mask，帮助模型在处理输入时区分出哪些token是实际数据（非填充值）。
    float(i > 0): 产生掩码，[1.0]代表真实的位置，[0.0]代表填充值的位置。
    '''
    train_masks = [[float(i > 0) for i in ii] for ii in train_tokens_ids]

    # to tensors
    # train_tokens_tensor, train_masks_tensor
    return torch.tensor(train_tokens_ids), torch.tensor(train_masks)



def to_dataloader(texts, extras, ys,
                  tokenizer,
                  max_seq_length,
                  batch_size,
                  dataset_cls = TensorDataset,
                  sample_cls = RandomSampler
                ):
    """
        Convert raw input into PyTorch dataloader
    """
    # Labels
    train_y_tensor = torch.tensor(ys).float()
    
    
    if texts is not None and extras is not None:
        # 使用 Full featrues 进行训练
        train_tokens_tensor, train_masks_tensor = text_to_train_tensors(texts, tokenizer, max_seq_length)
        train_extras_tensor = torch.tensor(extras, dtype=torch.float)
        
        train_dataset = dataset_cls(train_tokens_tensor, train_masks_tensor, train_extras_tensor, train_y_tensor) 

    elif texts is not None and extras is None:
        # 使用  Text only 进行训练
        train_tokens_tensor, train_masks_tensor = text_to_train_tensors(texts, tokenizer, max_seq_length)
        train_dataset = dataset_cls(train_tokens_tensor, train_masks_tensor, train_y_tensor)

    elif texts is None and extras is not None:  
        # 使用  extras only 进行训练
        train_extras_tensor = torch.tensor(extras, dtype=torch.float)
        train_dataset =dataset_cls(train_extras_tensor, train_y_tensor)
    else:
        raise ValueError("texts and extras cannot be both None")


    train_sampler = sample_cls(train_dataset)
    
    return DataLoader(dataset = train_dataset, sampler=train_sampler, batch_size=batch_size)



def get_extras_gender(df, extra_cols, author2vec, 
                      author2gender, with_vec=True, 
                      with_gender=True, on_off_switch=False):
        """
            Build matrix for extra data (i.e. author embeddings + gender)
            
            最后， extras[i] = meta_data + author_embedding + gender
            
            其中， meta_data = default_extra_cols
            
            
            return extras, vec_found_count, gender_found_count, 
                    vec_found_selector, gender_found_selector
        """
        
        
        if with_vec:
            '''
             iter(author2vec.values())：创建一个迭代器，遍历 author2vec 的所有值（即各个作者的向量）。
            '''
            AUTHOR_DIM = len(next(iter(author2vec.values())))
            
            if on_off_switch:
                AUTHOR_DIM += 1  # One additional dimension of binary (1/0) if embedding is available
        else:
            AUTHOR_DIM = 0
        
        
        if with_gender:
            GENDER_DIM = len(next(iter(author2gender.values())))
        else:
            GENDER_DIM = 0
        
        
        
        
        # extra_cols 很可能是一个字段列表：['year', 'month', 'day', 'author_id']
        extras = np.zeros((len(df), len(extra_cols)+AUTHOR_DIM+GENDER_DIM))
        
        # 复制[False]那么多的次数从而创建一个新的列表，其长度与 df 的行数相同。
        # vec_found_selector[i] 可以用于标识 df 中第 i 行的向量是否已被找到或处理。
        vec_found_selector = [False]*len(df)
        
        gender_found_selector = [False]*len(df)
        
        vec_found_count= 0
        gender_found_count = 0

    
        for i, authors in enumerate(df['authors']):
            
            # 先把meta-data装进去
            extras[i][:len(extra_cols)] = df[extra_cols].values[i]
            
            
            
            # 再把 author embedding 装进去
            if with_vec:
                for author in authors.split(";"):
                    if author in author2vec:
                        if on_off_switch:
                            extras[i][len(extra_cols):len(extra_cols) + AUTHOR_DIM -1] = author2vec[author]
                            
                            # 单独给switch留出一位，赋值为0/1, 表示该作者的向量是否被找到
                            extras[i][len(extra_cols) + AUTHOR_DIM] = 1 
                        else:
                            extras[i][len(extra_cols):len(extra_cols) + AUTHOR_DIM] = author2vec[author]
                    
                    vec_found_count+=1
                    vec_found_selector[i] = True
            
            
            
            
            # 再把author gender装进去
            if with_gender:
                for author in authors.split(";"):
                    first_name = author.split(" ")[0]
                    
                    if first_name in author2gender:
                        extras[i][len(extra_cols) + AUTHOR_DIM:]  = author2gender[first_name]
                        gender_found_count+=1
                        gender_found_selector[i] = True
                        break
                    
        return extras, vec_found_count, gender_found_count, vec_found_selector, gender_found_selector


def get_best_thresholds(labels, test_y, outputs, plot=False):
    """
        Hyper parameter search for best classification threshold
        
        labels； 预测的标签集合
        
        test_y: 真实标签[batch_size, num_labels]
        
            return t_max, f_max
            
            t_max: 每个标签对应的最佳阈值
            f_max: 每个标签对应的最佳f-score
    """
    t_max = [0] * len(labels) # contains best threshold for each label
    f_max = [0] * len(labels) # contains best f-score for each label

    for i, label in enumerate(labels):
        ts = []
        fs = []

        for t in np.linspace(0.1, 0.99, num=50):
            # 将模型输出 outputs[:,i] 应用阈值处理，将大于等于 t 的值设为1，其余设为0，得到二分类结果。
            p, r, f, _ = precision_recall_fscore_support(test_y[:,i], np.where(outputs[:,i]>t, 1, 0), average='micro')
            ts.append(t) # 当前标签对应的阈值列表
            fs.append(f) # 当前标签对应的f-score列表
            
            if f > f_max[i]: # 记录最好的f-score 以及 对应的阈值
                f_max[i] = f
                t_max[i] = t
            
        if plot:
            print(f'LABEL: {label}')
            print(f'f_max: {f_max[i]}')
            print(f't_max: {t_max[i]}')

            plt.scatter(ts, fs)
            plt.show()
        
        
    return t_max, f_max





def nn_output_to_submission(first_line, df, outputs, output_ids, t_max, labels, most_popular_label):
    """
        Convert BERT-output into submission format (only a single task)
        
        用每个标签的最佳阈值来过滤所有样本
        
        return lines, no_labels
        
            lines: 存储所有样本的预测标签，格式为：isbn\tpred_label1;pred_label2;pred_label3
            no_labels: 没有一个预测的标签达到阈值的样本的数量
    """
    
    # 没有一个预测的标签达到阈值的样本的数量
    no_label = 0
    
    lines = [first_line] # 存储表头行
    
    
    for idx in output_ids:
        pred_labels = [] # 存储第idx个样本的所有预测标签---有多个
        
        for i, label in enumerate(labels):
            if outputs[idx][i]>= t_max[i]:
                pred_labels.appned(label)
        
        
        
        if len(pred_labels)==0: # 连一个达到阈值的标签都没有
            no_labels+=1
            
            # If no label was predicted -> just use most popular
            pred_labels = most_popular_label
        else:
            pred_labels = ';'.join(pred_labels)
            
            
            
        # 取isbn
        isbn = df["isbn"].values[idx]
        
        lines.append(f'{isbn}\t{pred_labels}')
            
    return lines, no_labels


