
"""

Run experiments for paper results
        - for each experiment: sub-task A + B
- save model weights to disk
- save val results to disk

/experiments/
        bert_extras/
                weights/
                config/
                validation/
                stats/


export BERT_MODELS_DIR="/home/mostendorff/datasets/BERT_pre_trained_models/pytorch"

python cli.py run_on_val <name> $GPU_ID $EXTRAS_DIR $TRAIN_DF_PATH $VAL_DF_PATH $OUTPUT_DIR --epochs 5

    4 \
    data/extras \
    germeval_train_df_meta.pickle \
    germeval_val_df_meta.pickle \
    experiments_output

python experiments.py run task-a__bert-german_manual_author-embedding_author-gender \
    4 \
    data/extras \
    germeval_train_df_meta.pickle \
    germeval_val_df_meta.pickle \
    experiments_output


python experiments.py run task-a__bert-german_manual_no-embedding_author-gender \
    4 \
    data/extras \
    germeval_train_df_meta.pickle \
    germeval_val_df_meta.pickle \
    experiments_output
    
python experiments.py run task-a__bert-german_manual_no-embedding_author-gender \
    4 \
    data/extras \
    germeval_train_df_meta.pickle \
    germeval_val_df_meta.pickle \
    experiments_output


python experiments.py run task-a__bert-german_text-only \
    4 \
    data/extras \
    germeval_train_df_meta.pickle \
    germeval_val_df_meta.pickle \
    experiments_output

python experiments.py run task-a__author-only \
    4 \
    data/extras \
    germeval_train_df_meta.pickle \
    germeval_val_df_meta.pickle \
    experiments_output

python experiments.py run task-b__author-only \
    4 \
    data/extras \
    germeval_train_df_meta.pickle \
    germeval_val_df_meta.pickle \
    experiments_output

python experiments.py run task-b__bert-german_full \
    4 \
    data/extras \
    germeval_train_df_meta.pickle \
    germeval_val_df_meta.pickle \
    experiments_output \
    --epochs 5

----

python experiments.py final task-a__bert-german_full \
    4 \
    data/extras \
    germeval_fulltrain_df_meta.pickle \
    germeval_test_df_meta.pickle \
    experiments_output \
    --epochs 1

python experiments.py final task-b__bert-german_full     3     data/extras     germeval_fulltrain_df_meta.pickle     germeval_test_df_meta.pickle     experiments_output     --epochs 5


python experiments.py final task-a__bert-german_text-only     2     data/extras     germeval_fulltrain_df_meta.pickle     germeval_test_df_meta.pickle     experiments_output     --epochs 5


python experiments.py final task-b__bert-german_full     2     data/extras     germeval_fulltrain_df_meta.pickle     germeval_test_df_meta.pickle     experiments_output     --epochs 5

"""





import json
import os
import pickle
import numpy as np

import fire
import torch
import logging

from torch.optim import Adam
from sklearn.metrics import classification_report

# 从配置文件导入实验的超参数
# author dim: author-only-model 的hidden_size
from config import AUTHOR_DIM, LEARNING_RATE, TASK_A_LABELS_COUNT, TASK_B_LABELS_COUNT, most_popular_label
from data_utils import get_best_thresholds, nn_output_to_submission
from experiment import Experiment

# 导入MLP双层分类器
from models import LinearMultiClassifier

logging.basicConfig(level=logging.INFO)



# 定义实验 sub-task-A or B, bert+author embedding, meta data feature? 定义了用什么{模型+输入数据}的配置

experiments = {

  # ============== A ========================
    
  # full 代表 meta-data(author_gender+manual) + author_embedding(author_vec) 全部都输入
  
    'task-a__bert-german_full': Experiment(
        'a', 'bert-base-german-cased', with_text=True, with_author_gender=True, with_manual=True, with_author_vec=True
    ),

  # 2 代表两层MLP分类器
    'task-a__bert-german_full_2': Experiment(
        'a', 'bert-base-german-cased', with_text=True, with_author_gender=True, with_manual=True, with_author_vec=True, mlp_dim=500,
    ),

  # no-embedding 表示不要author-embedding
    'task-a__bert-german_manual_no-embedding': Experiment(
        'a', 'bert-base-german-cased', with_text=True, with_author_gender=True, with_manual=True, with_author_vec=False
    ),
  
  # 表示不要meta-data
    'task-a__bert-german_no-manual_embedding': Experiment(
        'a', 'bert-base-german-cased', with_text=True, with_author_gender=False, with_manual=False, with_author_vec=True
    ),

  # 除了标题和简介，其他都不要
    'task-a__bert-german_text-only': Experiment(
        'a', 'bert-base-german-cased', with_text=True, with_author_gender=False, with_manual=False, with_author_vec=False
    ),
    # author only model: 作者专门为author_embedding向量创建的分类器
    'task-a__author-only': Experiment(
        'a', '-', with_text=False, with_author_gender=False, with_manual=False, with_author_vec=True,
        classifier_model=LinearMultiClassifier(
            labels_count=TASK_A_LABELS_COUNT,
            extras_dim=AUTHOR_DIM,
        )
    ),
  
    # bert-base-multilingual-cased
    'task-a__bert-multilingual_text-only': Experiment(
        'a', 'bert-base-multilingual-cased', with_text=True, with_author_gender=False, with_manual=False, with_author_vec=False
    ),






  # =================== B =====================

'task-b__bert-german_full': Experiment(
        'b', 'bert-base-german-cased', with_text=True, with_author_gender=True, with_manual=True, with_author_vec=True
    ),
    'task-b__bert-german_manual_no-embedding': Experiment(
        'b', 'bert-base-german-cased', with_text=True, with_author_gender=True, with_manual=True, with_author_vec=False
    ),
    'task-b__bert-german_no-manual_embedding': Experiment(
        'b', 'bert-base-german-cased', with_text=True, with_author_gender=False, with_manual=False, with_author_vec=True
    ),
    'task-b__bert-german_text-only': Experiment(
        'b', 'bert-base-german-cased', with_text=True, with_author_gender=False, with_manual=False, with_author_vec=False
    ),
    # author only
    'task-b__author-only': Experiment(
        'b', '-', with_text=False, with_author_gender=False, with_manual=False, with_author_vec=True,
        classifier_model=LinearMultiClassifier(
            labels_count=TASK_B_LABELS_COUNT,
            extras_dim=AUTHOR_DIM,
        )
    ),
    # bert-base-multilingual-cased
    'task-b__bert-multilingual_text-only': Experiment(
        'b', 'bert-base-multilingual-cased', with_text=True, with_author_gender=False, with_manual=False, with_author_vec=False
    ),

    ######

    # switch does not work (??)
    'task-a__bert-german_full-switch': Experiment(
        'a', 'bert-base-german-cased', with_text=True, with_author_gender=True, with_manual=True, with_author_vec=True,
        author_vec_switch=True,
    ),


}




def run_on_val(name, cuda_device, extras_dir, df_train_path, df_val_path, 
                    output_dir, epochs=None, continue_training=False,
                    batch_size=None):

    if name not in experiments:
        print(f'该实验设置不存在~~~: {name}')
        exit(1)
        
    experiment: Experiment = experiments[name]
    experiment.name = name
    experiment.output_dir = output_dir
    
    experiment.init(cuda_device, epochs, batch_size, continue_training)
    
    
    train_dataloader, \
        val_dataloader,\
        vec_found_selector, \
            val_df, val_y = experiment.prepare_data_loaders(df_train_path, df_val_path, extras_dir)
    
    
    
    model = experiment.get_model()
    
    print(f"当前使用的模型是：{type(model).__name__}")
    
    
    
    # Load existing model weights
    if continue_training:
        print('Loading existing model weights...')
        model.load_state_dict(torch.load(os.path.join(experiment.get_output_dir(), 'model_weights')))


    # Training
    
    optimizer = Adam(model.parameters(), lr = LEARNING_RATE)
    
    # Model to GPU
    model = model.cuda()
    
    experiment.train(model, optimizer, train_dataloader)
    
    
    
    # Validation 
    output_ids, outputs= experiment.eval(model, val_dataloader)
    
    
    # val_y : [batch_size , num_labels]
    # outputs: [batch_size , num_labels]
    t_max, f_max = get_best_thresholds(experiment.labels, val_y, outputs, plot=False)
    
    
    report = classification_report(val_y, np.where(outputs>=t_max, 1, 0), \
                                   target_names=experiment.labels, output_dict=True)
    
    report_str = classification_report(val_y, np.where(outputs>=t_max, 1, 0), \
                                target_names=experiment.labels, output_dict=False)   

    
    if vec_found_selector is not None and len(vec_found_selector)>0:
        try:
            report_author_vec = classification_report(val_y[vec_found_selector], np.where(outputs[vec_found_selector]>=t_max, 1, 0), \
                                   target_names=experiment.labels, output_dict=True)
            
            report_author_vec_str = classification_report(val_y[vec_found_selector], np.where(outputs[vec_found_selector]>=t_max, 1, 0), \
                                target_names=experiment.labels, output_dict=False)  
            
        except:
            print("Can not report author vec found")
    
    
    
    # Save
    
    # 将字典写入json
    with open(os.path.join(experiment.get_output_dir(), 'report.json'), 'w') as f:
        json.dump(report, f)
    
    # 将文本写入txt
    with open(os.path.join(experiment.get_output_dir(), 'report.txt'), 'w') as f:
        f.write(report_str)
        
        
        
    if vec_found_selector is not None and len(vec_found_selector)>0:
        try:
            with open(os.path.join(experiment.get_output_dir(), "report_author_vec_found.json"), 'w') as f:
                json.dump(report_author_vec, f)
                
            with open(os.path.join(experiment.get_output_dir(), "report_author_vec_found.txt"), 'w') as f:
                f.write(report_author_vec_str)
                
        except BaseException:
            print('Cannot write report_author_vec_found')
    
    
    # 将最佳阈值写入文件
    with open(os.path.join(experiment.get_output_dir(), 'best_thresholds.csv'), 'w') as f:
        f.write(','.join([str(t) for t in t_max]))
        
    # 将模型预测的logits和对应的ids写入文件
    with open(os.path.join(experiment.get_output_dir(), 'outputs_with_ids.pickle'), 'wb') as f:
        pickle.dump((outputs, output_ids), f)
    
    # 存储模型权重
    torch.save(model.state_dict(), os.path.join(experiment.get_output_dir(), 'model_weights'))

    # 将模型配置写入文件
    with open(os.path.join(experiment.get_output_dir(), 'model_config.json'), 'w') as f:
        json.dump(model.config, f)
        
    
    # Submission
    lines, no_label = nn_output_to_submission("subtask_" + experiment.task,
                                                val_df, outputs, output_ids, t_max, 
                                                experiment.labels, most_popular_label)
    
    print(f"没有预测出标签的样本有多少？{no_label}")
    
    fn = os.path.join(experiment.get_output_dir(), 'submission.txt')
    with open(fn, 'w') as f:
        f.write('\n'.join(lines))
        
    print(f'Submission file saved to: {fn}')
    
    
    
    
    

def run_on_test(name, cuda_device, extras_dir, df_full_path, df_test_path, output_dir, epochs=None, continue_training=False,
        batch_size=None):

    if name not in experiments:
        print(f'Experiment not found: {name}')
        exit(1)

    experiment = experiments[name]
    experiment.name = 'final-' + name
    experiment.output_dir = output_dir
    experiment.init(cuda_device, epochs, batch_size, continue_training)

    # best thresholds from validation set
    t_fn = os.path.join(output_dir, name, 'best_thresholds.csv')
    if not os.path.exists(t_fn):
        raise ValueError('Could not load threshold values')

    train_dataloader, test_dataloader, vec_found_selector, test_df, _ = experiment.prepare_data_loaders(df_full_path, df_test_path, extras_dir, test_set=True)

    # Parse thresholds
    with open(t_fn, 'r') as f:
        t_max = [float(t) for t in f.read().split(',')]

        if len(t_max) != len(experiment.labels):
            raise ValueError('Threshold values does not match label count')

    model = experiment.get_model()

    print(f'Using model: {type(model).__name__}')

    # Load existing model weights
    if continue_training:
        print('Loading existing model weights...')
        model.load_state_dict(torch.load(os.path.join(experiment.get_output_dir(), 'full_model_weights')))

    # Training
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # Model to GPU
    model = model.cuda()

    experiment.train(model, optimizer, train_dataloader)

    # Save trained model
    torch.save(model.state_dict(), os.path.join(experiment.get_output_dir(), 'model_weights'))
    with open(os.path.join(experiment.get_output_dir(), 'model_config.json'), 'w') as f:
        json.dump(model.config, f)

    # Test results
    output_ids, outputs = experiment.eval(model, test_dataloader)

    # Store predictions
    with open(os.path.join(experiment.get_output_dir(), 'outputs_with_ids.pickle'), 'wb') as f:
        pickle.dump((outputs, output_ids), f)

    # Submission
    lines, no_label = nn_output_to_submission('subtask_' + experiment.task, test_df, outputs, output_ids, t_max, experiment.labels,
                                              most_popular_label)
    print(f'-- no found: {no_label}')

    fn = os.path.join(experiment.get_output_dir(), 'submission.txt')
    with open(fn, 'w') as f:
        f.write('\n'.join(lines))

    print(f'Submission file saved to: {fn}')




if __name__ == '__main__':
    fire.Fire()
