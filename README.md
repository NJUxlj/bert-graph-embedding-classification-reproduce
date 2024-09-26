## 复现并改进一篇论文：Enriching BERT with Knowledge Graph Embeddings for Document Classification

### 原项目地址
[pytorch-bert-document-classification](https://github.com/malteos/pytorch-bert-document-classification)

### 论文地址：
[Enriching BERT with Knowledge Graph Embeddings for Document Classification](https://arxiv.org/abs/1909.08402)


### 项目内容
- 本项目主要研究使用简短描述性文本（封面简介）和附加元数据对书籍进行分类。

- 作者演示了如何将文本表示（text-representation）与元数据(meta-data)和知识图嵌入(knowledge-graph-embeddings)相结合，并使用上述手段对作者信息进行编码。
- 评测集：2019 GermEval shared task on hierarchical text classification （即，共享任务数据集）
- 我们采用BERT进行文本分类，并在共享任务的上下文中提供额外的元数据来扩展模型，如作者、出版商、出版日期等。

- 注意，作者仅仅使用了书的简介来作为分类模型的输入。


### 项目主要贡献
- 使用最先进的文本处理方法纳入了额外的（元数据）数据。

### 数据集和任务
- 2019 GermEval shared task on hierarchical text classification
- 一个包含了20784本德语书的数据集
- 每行记录 = {title, author_list, blurb简介, 书籍URL, ISBN, 出版日期}
- 所有书使用了德国出版商兰登书屋使用的分类法进行标记。{一级类别:8个类别, 二级类别:93， 三级类别:242}
- 每个共享任务 = 子任务A + 子任务B.
- 子任务A：是一个多标签分类任务，从8个类别中选出多个
- 子任务B：也是多标签分类任务，从二级+三级的总共343个类别中选出多个。

### 原文模型结构
1. 新模型 = 原始Bert模型+双层MLP分类器+输出层softmax
2. 输入数据分为两种：文本特征、非文本特征。
3. 文本特征：标题+简介，不超过300 tokens
4. 非文本特征包含：{元数据特征(meta-data feature：10维向量（其中有2维代表性别）、作者信息嵌入(author embedding)：200维向量}
5. gender字段取值：Probability of first author being male or female based on the **Gender-by-Name** dataset
6. Bert模型的输入格式 = 书籍标题+书籍简介，
7. MLP分类器的输入：bert的输出+元数据特征+作者信息嵌入
![image](https://github.com/user-attachments/assets/9bc7a6a2-bf28-49c6-97be-5bbaf3d401e9)


### 训练过程
```python
batch_size b= 16
dropout probability d = 0.1
learning rate η = 2−5 (Adam optimizer)
training epochs = 5

所有实验都在GeForce GTX 1080 Ti（11 GB）上运行，因此单个训练周期最多需要10分钟。
如果没有一个标签的预测概率高于分类阈值，则使用最流行的标签（比如：文学和非文化）作为预测。
```

### 如何运行原项目

####
命令行环境变量
- TRAIN_DF_PATH: Path to Pandas Dataframe (pickle)
- GPU_ID: Run experiments on this GPU (used for CUDA_VISIBLE_DEVICES)
- OUTPUT_DIR: Directory to store experiment output
- EXTRAS_DIR: Directory where author embeddings and gender data is located
- BERT_MODELS_DIR: Directory where pre-trained BERT models are located

Validation set
```shell
python cli.py run_on_val <name> $GPU_ID $EXTRAS_DIR $TRAIN_DF_PATH $VAL_DF_PATH $OUTPUT_DIR --epochs 5
```

Test set
```shell
python cli.py run_on_test <name> $GPU_ID $EXTRAS_DIR $FULL_DF_PATH $TEST_DF_PATH $OUTPUT_DIR --epochs 5
```

Evaluation
```shell
The scores from the result table can be reproduced with the evaluation.ipynb notebook.
```

### 原实验结果
- 使用额外的元数据特征和作者信息嵌入的BERT-German的设置优于所有其他设置
- 任务A的F1得分为87.20，任务B的F1得分是64.70
![image](https://github.com/user-attachments/assets/664fbea3-fcc1-4c0b-bced-862b910bd911)



### 我做出了如下改进
- 由于阿里云DSW实例的网络问题，我无法访问huggingface。
- 因此使用bert-base-chinese 代替 bert-base-german
#### 改进1
- 原始项目使用了德语写的书籍的简介作为语料，并且最终预训练了一个德语模型。具体来说，该模型是在德国维基百科、新闻文章和法院判决书上从头开始训练的。
- 我将训练语料改为中文，并且想输入格式与国内习惯相匹配。
- **已抛弃**


#### 改进2
共享数据集的分类法无法找到中文的相似的版本，因此我暂时只能在 text-feature, extra featrue 这两块特征的组合上做文章

#### PCA


#### forward featrue selection 




#### 改进3
- 我们改进模型的架构
- 原始模型采用了 BERT + MLP + softmax

  - 原始的BERT论文提出，在从左到右模型(LTR)顶上加上BiLSTM层后，性能仍然无法超过BERT, 同时论文提出并证明BiLSTM会伤害GLUE数据集上的表现
  - 另外，在某些信息来源中，BERT+BiLSTM的表现甚至不如原始BERT
  - 我们分析， 在一些文本分类任务重， 双向编码可能并不是必要的，可能会产生对目标token的过度预测，因此有必要实验原始的LSTM， 同时保留语序信息。
  - 另外， TextCNN在pooling时也能有效保留语序信息。

- 因此，我提出要使用
BERT + TextRCNN(LSTM+TextCNN) + MLP + SoftMAX


```python

class ExtraBertTextRCNNMultiClassifier(nn.Module):
    def __init__(self, bert_model_path, labels_count, rnn_hidden_dim=256, cnn_kernel_size=3,   
                 rnn_type='LSTM', num_layers=1, hidden_dim=768, mlp_dim=256, extras_dim=6, dropout=0.1):  
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
        self.rnn = nn.LSTM(input_size=hidden_dim, hidden_size=rnn_hidden_dim,  
                           num_layers=num_layers, batch_first=True, bidirectional=True)  

        # 初始化CNN层用于特征提取  
        self.cnn = nn.Conv1d(in_channels=2 * rnn_hidden_dim,  # 因为是双向LSTM  
                             out_channels=rnn_hidden_dim,  
                             kernel_size=cnn_kernel_size,  
                             padding=cnn_kernel_size // 2)  

      
        self.dropout = nn.Dropout(dropout)  

        # MLP全连接层  
        self.mlp = nn.Sequential(  
            nn.Linear(rnn_hidden_dim + extras_dim, mlp_dim),  
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
        extras: [batch_size, extras_dim]  
        '''  
        # 通过BERT模型获取最后一个隐藏层的输出和池化输出  
        sequence_output, _ = self.bert(tokens, attention_mask=masks, return_dict=False)  
        
        # 通过RNN  
        rnn_output, _ = self.rnn(sequence_output)  

        # 转换维度使其适应卷积层 (batch_size, channels, seq_len)  
        rnn_output = rnn_output.permute(0, 2, 1)  

        # 通过卷积层提取特征  
        cnn_output = self.cnn(rnn_output)  

        # 转换卷积输出维度  
        cnn_output = cnn_output.permute(0, 2, 1)  

        # 提取最后一个时间步的特征作为文本特征  
        text_features = torch.mean(cnn_output, dim=1)  

        dropout_output = self.dropout(text_features)  

        concat_output = torch.cat([dropout_output, extras], dim=1)  

        mlp_output = self.mlp(concat_output)  

        # 输出概率  
        proba = self.softmax(mlp_output)  

        return proba  



```

- 后期，我们也会加入Diy的BERT（纯numpy实现）


#### 改进4
- 根据论文的说法，在所有的实验设置中（sub-taskA,B + text-feature or not + extra-features or not), 
模型的精确率都显著高于召回率。
- 作者认为，对于子任务B中的343个标签中的一些，实例很少。这意味着，如果分类器预测某个标签，它很可能是正确的（即高精度），但对于许多具有低频标签的情况，这个低频标签永远不会被预测（即低召回率）。🔤

##### 解决标签不均衡问题
1.**过采样** 复制指定类别的样本，在采样中重复
2.**降采样** 减少多样本类别的采样，随机使用部分



### 实验结果


### 准备模型

- 请根据任务类型自行下载对应的模型

(final-task-a__bert-german_full.tar.gz)[https://github.com/malteos/pytorch-bert-document-classification/releases]




### 准备数据
---

#### GermEval data
- Download from shared-task website: (here)[https://competitions.codalab.org/competitions/20139]
- Run all steps in Jupyter Notebook: \underline{germeval-data.ipynb}

#### Author Embeddings
- (Download pre-trained Wikidata embedding (30GB): Facebook PyTorch-BigGraph)[https://github.com/facebookresearch/PyTorch-BigGraph#pre-trained-embeddings]
- (Download WikiMapper index files (de+en))[https://github.com/jcklie/wikimapper#precomputed-indices]

### GermEval数据集有啥？
![alt text](image.png)
- The dataset consists of information on German books (blurb, genres, title, author, URL, ISBN, date of publication), crawled from the Random House page.

```shell
Number of books	20784
Average Number of Words	94.67
Number of classes	343 (8, 93, 242 on level 1,2, 3 resp.)

```


- A sample entry is shown below:
```html
<book date="2019-01-04" xml:lang="de">
<title>Blueberry Summer</title>
<body>In neuer Sommer beginnt für Rory und Isabel – mit einer kleinen Neuerung: Rory ist fest mit Connor Rule zusammen und deshalb als Hausgast in den Hamptons ... Und Isabel wieder auf Mike.</body>
<copyright>(c) Penguin Random House</copyright>
<categories>
<category>
<topic d="0">Kinderbuch & Jugendbuch </topic>
<topic d="1" label="True">Liebe, Beziehung und Freundschaft</topic>
</category>
<category>
<topic d="0">Kinderbuch & Jugendbuch </topic>
<topic d="1" label="True">Echtes Leben, Realistischer Roman</topic>
</category>
</categories>
<authors>Joanna Philbin</authors>
<published>2015-02-09</published>
<isbn>9780451457998</isbn>
<url>https://www.randomhouse.de/Taschenbuch/Blueberry-Summer/Joanna-Philbin/cbj-Jugendbuecher/e455949.rhd%0A/</url>
</book>


```

### 什么是Pytorch-BigGraph?
    1. PyTorch-BigGraph (PBG) is a distributed system for learning graph embeddings for large graphs, particularly big web interaction graphs with up to billions of entities and trillions of edges.
   
    2. PBG trains on an input graph by ingesting its list of edges, each identified by its source and target entities and, possibly, a relation type. It outputs a feature vector (embedding) for each entity, trying to place adjacent entities close to each other in the vector space, while pushing unconnected entities apart. Therefore, entities that have a similar distribution of neighbors will end up being nearby.
   
    3. 简单来说， pytorch-BigGraph是一套标准的模型训练系统，他允许你使用`大型知识图谱`来训练模型，训练好的模型可以为知识图谱中的每个实体生成一个向量。不同向量之间的远近距离代表对应实体之间的关系。

    4. 因此， 我们的 Wikidata graph， 就是一种 BigGraph


### 什么是Wikidata?
- 就是已经被Pytorch-BigGraph训练好的实体(entity)和边(edge\relation)的嵌入文件
- 大致包含两个文件：
  - `entity_embeddings.tsv`
  - `relation_types_enbeddings.tsv`

### 什么是WikiMapper？
   1. This small Python library helps you to map Wikipedia page titles (e.g. Manatee to Q42797) and vice versa. This is done by creating an index of these mappings from a Wikipedia SQL dump. Precomputed indices can be found under Precomputed indices. Redirects are taken into account.
   2. 简单来说，就是先去wikimedia下载一个维基百科索引数据库文件(enwiki-latest.db)
   3. 然后wikimapper这个软件，通过索引文件，把wikipedia的标题映射到之前wikidata中的id (实体id或关系id)


#### Map Wikipedia page title to Wikidata id
```python
from wikimapper import WikiMapper

mapper = WikiMapper("index_enwiki-latest.db")
wikidata_id = mapper.title_to_id("Python_(programming_language)")
print(wikidata_id) # Q28865
```


### 使用WikiMapper创建Wiki索引

```shell
pip install wikimapper

wikimapper download enwiki-latest --dir wikidata

wikimapper create enwiki-latest --dumpdir wikidata --target wikidata/index_enwiki-latest.db
```

### 使用wikimapper索引和wikidata 创建author-embedding
```shell
!python wikidata_for_authors.py run ./datasets/wikidata/index_enwiki-latest.db \
    ./datasets/wikidata/index_dewiki-latest.db \
    ./datasets/wikidata/torchbiggraph/wikidata_translation_v1.tsv.gz \
    ./eval/authors.pickle \
    ./eval/author2embedding.pickle
    

# /eval/author2embedding.pickle 是输出路径

```


### Reference
```latex
@inproceedings{Ostendorff2019,
    address = {Erlangen, Germany},
    author = {Ostendorff, Malte and Bourgonje, Peter and Berger, Maria and Moreno-Schneider, Julian and Rehm, Georg},
    booktitle = {Proceedings of the GermEval 2019 Workshop},
    title = {{Enriching BERT with Knowledge Graph Embedding for Document Classification}},
    year = {2019}
}


```
