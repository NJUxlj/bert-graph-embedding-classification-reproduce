import os

# Global settings
# (same for all experiments)

MAX_SEQ_LENGTH = 300 #256  # max 512 (strongly affects GPU memory consumption)
HIDDEN_DIM = 768  # size of BERT hidden layer
MLP_DIM = 1024 #500  # size of multi layer perceptron (2 layers)
AUTHOR_DIM = 200  # size of Wikidata author embeddings
GENDER_DIM = 2
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 5
TASK_A_LABELS_COUNT = 8  # 一级标签数量 
TASK_B_LABELS_COUNT = 343   # 二级+三级标签总数
default_extra_cols = [    # meta data features
    'author_count',
    'authors_academic',    # 学术头衔{Dr., Prof.}
    'title_words', 'text_words', # 标题字数，简介字数
    'age',
    'word_len_max',
    'word_len_mean',
    'word_len_median'
]
most_popular_label = 'Literatur & Unterhaltung'  # use this as default

if 'BERT_MODELS_DIR' not in os.environ:
    raise ValueError('You must define BERT_MODELS_DIR as environment variable!')

BERT_MODELS_DIR = os.environ['BERT_MODELS_DIR']
