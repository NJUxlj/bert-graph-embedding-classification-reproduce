
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler


from tqdm import tqdm

# 用于序列化和反序列化Python对象。序列化是将对象转换为字节流的过程
# pickle.dump, pickle.load
import pickle

from pytorch_pretrained_bert import BertTokenizer

from config import BERT_MODELS_DIR, MAX_SEQ_LENGTH, MLP_DIM, AUTHOR_DIM, GENDER_DIM, HIDDEN_DIM, \
    TRAIN_BATCH_SIZE,default_extra_cols, NUM_TRAIN_EPOCHS

from models import ExtraBertMultiClassifier, BertMultiClassifier, ExtraBertTextRCNNMultiClassifier

from data_utils import get_extras_gender, to_dataloader, TensorIndexDataset



class Experiment(object):
    """
    Holds all experiment information
    """
    name = None
    output_dir = None
    epochs = None
    batch_size = None
    device = None
    labels = None

    def __init__(self, task, bert_model, classifier_model=None, with_text=True, with_author_gender=True,
                 with_manual=True, with_author_vec=True, author_vec_switch=False, mlp_dim=None):
        self.task = task
        self.bert_model = bert_model
        self.with_text = with_text
        self.with_author_gender = with_author_gender
        self.with_manual = with_manual
        self.with_author_vec = with_author_vec
        self.author_vec_switch = author_vec_switch
        self.classifier_model = classifier_model

        self.mlp_dim = mlp_dim if mlp_dim is not None else MLP_DIM



    def init(self, cuda_device, epochs, batch_size, continue_training):
        # Set GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

        if not torch.cuda.is_available():
            print('CUDA GPU is not available')
            exit(1)

        self.epochs = epochs if epochs is not None else NUM_TRAIN_EPOCHS
        self.batch_size = batch_size if batch_size is not None else TRAIN_BATCH_SIZE

        if not continue_training and os.path.exists(self.get_output_dir()):
            print(f'Output directory exist already: {self.get_output_dir()}')
            exit(1)
        else:
            os.makedirs(self.get_output_dir())




    
    def get_output_dir(self):
        return os.path.join(self.output_dir, self.name)
        

    def get_bert_model_path(self):
        return os.path.join(BERT_MODELS_DIR, self.bert_model)



        
    def get_author_dim(self):
        # Use author switch?
        if self.author_vec_switch:
            author_dim = AUTHOR_DIM + 1
        else:
            author_dim = AUTHOR_DIM

        

    def get_extra_cols(self):
        '''
            EXTRA_DIR 存储了 author embedding， 以及 gender data
        '''
        if self.with_manual: # 如果输入包含元数据
            extra_cols = default_extra_cols
        else:
            extra_cols = []
        
        return extra_cols



    def prepare_data_loaders(self, df_train_path, df_val_path,extras_dir, test_set=False):
        '''
          return train_dataloader, val_dataloader, vec_found_selector, val_df, val_y
        
            val_df: 验证集的数据集
            val_y: 验证集的标签
            
            vec_found_selector： 一个布尔数组，表示数据集中的哪几个本书找到了 author embedding
        '''
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.with_text:
            tokenizer = BertTokenizer.from_pretrained(self.get_bert_model_path(), do_lower_case=False)
        else:
            tokenizer = None

        # load external data
        if self.with_author_vec:
            with open(os.path.join(extras_dir, 'author2embedding.pickle'), 'rb') as f:
                author2vec = pickle.load(f)

            print(f'Embeddings avaiable for {len(author2vec)} authors')
        else:  
            author2vec = None


        # load gender data
        if  self.with_author_gender:
            '''
            这段代码是用来加载和处理一个包含姓名和性别概率的数据集的代码

            字典推导式: 这是一个字典推导式，用来创建字典。字典的键是姓名，值是一个numpy数组。

                row['name']: 从行中提取'name'列的值，作为字典的键。

                np.array([row['probability'], 0] if row['gender'] == 'M' else [0, row['probability']])：

                使用np.array创建一个numpy数组。
                检查row['gender']是否等于'M'：
                如果是男性'M'，则数组为[row['probability'], 0]，表示男性的概率。
                否则为女性（非'M'），数组为[0, row['probability']]，表示女性的概率。
                最终的结果是一个字典author2gender，其中键是姓名，值是一个包含两个元素的数组，分别表示男性和女性的概率。
            '''

            gender_df = pd.read_csv(os.path.join(extras_dir, 'name_gender.csv'))

            author2gender = {
                row['name']:np.array([ row['probability'], 0]
                        if row['gender']=='M' else 
                        [0, row["probability"]])
                for idx, row in gender_df.iterrows()
            }
            print(f'Gender data avaiable for {len(author2gender)} authors')
        else:
            author2gender = None



        # load training data
        with open(df_train_path, 'rb') as f:
            train_df, doc_cols, task_b_labels, task_a_labels = pickle.load(f)

        # Define labels (depends on task)
        if self.task == 'a':
            self.labels = task_a_labels
        elif self.task == 'b':
            self.labels = task_b_labels
        else:
            raise ValueError('Invalid task specified')


        # 只要存在任何一种extra features
        if self.with_manual or self.with_author_gender or self.with_author_vec:
            train_extras, vec_found_count, gender_found_count, _, _ = get_extras_gender( 
                train_df,
                self.get_extra_cols(),
                author2vec,
                author2gender,
                # 标签：是否加入 author_embedding
                with_vec = self.with_author_vec,
                with_gender = self.with_author_gender,
                on_off_switch = self.author_vec_switch
            )
        else:
            train_extras = None

        # 合并书籍的标题和简介
        if self.with_text:
            train_texts = [t + '.\n' + train_df['text'].values[i] for i, t in enumerate(train_df['title'].values)]
        else:
            train_texts = None


        # 获取标签
        train_y = train_df[self.labels].values


        train_dataloader = to_dataloader(train_texts, train_extras, train_y,
                                        tokenizer,
                                        MAX_SEQ_LENGTH,
                                        self.batch_size,
                                        dataset_cls = TensorDataset,
                                        sample_cls= RandomSampler,
        )



        # load validation data
        with open(df_val_path, "rb") as f:
            val_df,_,_,_ = pickle.load(f)

        

        if self.with_manual or self.with_author_gender or self.with_author_vec:
            val_extras, vec_found_count, gender_found_count, vec_found_selector, gender_found_selector = get_extras_gender(
                val_df,
                self.get_extra_cols(),
                author2vec,
                author2gender,
                with_vec = self.with_author_vec,
                with_gender = self.with_author_gender,
                on_off_switch = self.author_vec_switch,
            )
        else:
            val_extras = None
            vec_found_selector = None



        if self.with_text:
            val_texts = [t + '.\n' + val_df['text'].values[i] for i, t in enumerate(val_df['title'].values)]
        else:
            val_texts = None
        

        # 
        if test_set:
            # 为每个测试集样本准备一个多标签预测模板，所有值初始化为0。
            val_y = np.zeros((len(val_texts), len(self.labels)))
        else:
            # val_df[self.labels]: 从Pandas数据框val_df中选取与self.labels字段列表对应的列。self.labels通常是一个标签名称的集合。
            # .values：将所选择的DataFrame部分转换为NumPy数组, 并取出所有行
            val_y = val_df[self.labels].values


        val_dataloader = to_dataloader(
            val_texts, val_extras, val_y, tokenizer,
            MAX_SEQ_LENGTH, self.batch_size,
            dataset_cls = TensorIndexDataset,
            sample_cls= SequentialSampler,
        )



        return train_dataloader, val_dataloader, vec_found_selector, val_df, val_y

        




    def get_model(self):
        if self.classifier_model is None:
            
            extras_dim = len(self.get_extra_cols()) # 首先获取 meta-data feature 列的数量

            if self.with_author_vec:
                extras_dim += self.get_author_dim()


            if self.with_author_gender:
                extras_dim += GENDER_DIM

            
            # 是否使用额外的信心进行分类

            if extras_dim > 0:
                model  = ExtraBertMultiClassifier(
                    bert_model_path=self.get_bert_model_path(),
                    labels_count=len(self.labels),
                    hidden_dim=HIDDEN_DIM,
                    mlp_dim=self.mlp_dim,
                    extras_dim=extras_dim,
                    dropout=self.dropout
                )





            # 使用纯文本分类
            else:
                model = BertMultiClassifier(
                    bert_model_path=self.get_bert_model_path(),
                    labels_count = len(self.labels),
                    hidden_dim = HIDDEN_DIM
                )


        else:
            model = self.classifier_model

        
        return model

            


        
    def train(self, model:nn.Module, optimizer, train_dataloader):
        for epoch_num in range(self.epochs):
            model.train()
            train_loss = 0
            
            print(f'Epoch: {epoch_num + 1}/{self.epochs}')

            # 显示进度条
            for step_num, batch_data in enumerate(tqdm(train_dataloader, desc='Iteration')):
                
                if self.with_text and (self.with_manual or self.with_author_gender or self.with_author_vec):
                    '''
                     只要存在extra features，那么需要将text和extra features合并在一起
                    '''
                    
                    # full featues
                    token_ids, masks, extras, gold_labels = tuple(t.to(self.device) for t in batch_data)
                    
                    probas = model(token_ids, masks, extras) # 调用ExtraBertMultiClassifier的forward方法  
                
                elif self.with_text:
                    
                    # text only 
                    token_ids, masks, gold_labels = tuple(t.to(self.device) for t in batch_data)
                    
                    probas = model(token_ids, masks) # 调用BertMuliClassifier的forward方法
                
                
                else:
                    # extras only
                    extras, gold_labels = tuple(t.to(self.device) for t in batch_data)
                    probas = model(extras)
                    
                
                loss_func = nn.BCELoss()
                batch_loss = loss_func(probas, gold_labels)
                train_loss += batch_loss.item() # 累加
                
                
                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
            print(f'\r第{epoch_num}轮 loss: {batch_loss.item()}')

            print(str(torch.cuda.memory_allocated(self.device) / 1000000) + 'M')
            

        return model


    def eval(self, model:nn.Module, dataloader):
        '''
         作用：计算模型在验证集上的性能，具体来说，
            计算val_df中每个样本的logits，并将所有logits，竖向拼接起来，
            得到一个二维数组outputs
            
            return output_ids, outputs
            
            outputs: [batch_size, labels_count]
            output_ids: [batch_size]
        '''
        model.eval()
        
        output_ids = []
        outputs = None
        
        
        with torch.no_grad():
            for step_num, batch_item in enumerate(dataloader):
                batch_ids, batch_data = batch_item
                
                if self.with_text and (self.with_author_gender or self.with_manual or self.with_author_vec):
                    token_ids, masks, extras, _   = tuple(t.to(self.device) for t in batch_data)
                    logits = model(token_ids, masks, extras)
                    
                elif  self.with_text:
                    token_ids, masks, _   = tuple(t.to(self.device) for t in batch_data)
                    logits = model(token_ids, masks)
                else:
                    # extras only
                    extras, _ = tuple(t.to(self.device) for t in batch_data)
                    logits = model(extras)
                
                # detach from GPU
                numpy_logits = logits.cpu().detach().numpy()
                
                
                if outputs is None: # 第一轮， 啥都没有
                    outputs = numpy_logits
                else:
                    np.vstack(outputs, numpy_logits)
                
                output_ids+=batch_ids.tolist()
                
        print(f"Evaluation completed for {len(outputs)} items")
                
        return output_ids, outputs
