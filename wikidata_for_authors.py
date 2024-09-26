"""

Prepare the Wikidata knowledge graph embeddings

# latest 
python wikidata_for_authors.py run ./datasets\wikidata\index_enwiki-latest.db \
    ./datasets/wikidata/index_dewiki-latest.db \
    ./datasets/wikidata/torchbiggraph/wikidata_translation_v1.tsv.gz \
    ./eval/authors.pickle \
    ./eval/author2embedding.pickle



# Extract embeddings for authors
python wikidata_for_authors.py run ~/datasets/wikidata/index_enwiki-20190420.db \
    ~/datasets/wikidata/index_dewiki-20190420.db \
    ~/datasets/wikidata/torchbiggraph/wikidata_translation_v1.tsv.gz \
    ~/notebooks/bert-text-classification/authors.pickle \
    ~/notebooks/bert-text-classification/author2embedding.pickle

Found 3684 QIDs for authors (not found: 11779)

# Convert for projector
python wikidata_for_authors.py convert_for_projector \
    ~/notebooks/bert-text-classification/author2embedding.pickle
    extras/author2embedding.projector.tsv \
    extras/author2embedding.projector_meta.tsv

"""



import pickle

import fire # 允许用户通过命令行参数直接操作函数或对象
import numpy as np

# 用于在 Wikipedia 标题和 Wikidata QID 之间进行映射。
# WikiMapper 使用一个索引文件来将 Wikipedia 页面（根据其标题）与相应的Wikidata实体（QID）关联起来
from wikimapper import WikiMapper

# 支持处理大文件， 远程文件， 文件IO
from smart_open import open


# 把作者名字映射到维基百科信息嵌入
def run(wikimapper_index_en, wikimapper_index_de, graph_embedding_file, authors_file, out_file):
        """

        Find the correct Wikidata embeddings for authors in `author_file` and write them
        into a author2embedding mapping file.

        :param wikimapper_index_en: 用于英文Wikipedia的wikimapper索引文件。
        :param wikimapper_index_de: 用于德文Wikipedia的wikimapper索引文件。
        :param graph_embedding_file: 存储图嵌入（graph embeddings）数据的文件。
        :param authors_file:
        :param out_file:
        :return:
        """
        
        
        print('Starting...')
        
        
        with open(authors_file, "rb") as f:
            authors_list =  pickle.load(f)

        print('Author file loaded')
        
        
        # 使用英文索引创建一个 WikiMapper 实例，用于将作者的Wikipedia英语标题映射到相应的Wikidata QID。
        en_mapper = WikiMapper(wikimapper_index_en) 
        de_mapper = WikiMapper(wikimapper_index_de)
        
        print('WikiMapper loaded (de+en)')
        
        not_found = 0 # 没找到id的 author数量
        
        not_found_ = []
        
        
        selected_entity_ids = set()
        found = 0
        
        qid2author = {}
        
        
        for book_authors_str in authors_list:
            authors = book_authors_str.split(';')
        
            for author in authors:
                
                qid = None
                
                # Wikipedia article might have the occupation in parenthesis
                en_queries = [
                    author,
                    author.replace(' ', '_'),
                    author.replace(' ', '_') + '_(novelist)',
                    author.replace(' ', '_') + '_(poet)',
                    author.replace(' ', '_') + '_(writer)',
                    author.replace(' ', '_') + '_(author)',
                    author.replace(' ', '_') + '_(journalist)',
                    author.replace(' ', '_') + '_(artist)',
                ]
                for query in en_queries:  # Try all options
                    qid = en_mapper.title_to_id(query)
                    if qid is not None: # 取到当前author的一条信息即可
                        break
                    
                
                if qid is None: # Try German version
                    de_queries = [
                        author,
                        author.replace(' ', '_'),
                        author.replace(' ', '_') + '_(Dichter)',
                        author.replace(' ', '_') + '_(Schriftsteller)',
                        author.replace(' ', '_') + '_(Autor)',
                        author.replace(' ', '_') + '_(Journalist)',
                        author.replace(' ', '_') + '_(Autorin)',
                    ]
                    for query in de_queries:  # Try all options
                        qid = de_mapper.title_to_id(query)
                        if qid is not None:
                            break
                
                if qid is None:
                    not_found+=1
                    not_found_.append(author)
                    
                else:
                    found+=1
                    selected_entity_ids.add(qid)
                    qid2author[qid] = author
                    
                
            print(f'Found {len(selected_entity_ids)} QIDs for authors (not found: {not_found})')


            # 作者名到Wikipedia嵌入的映射
            # 将特定的Wikidata实体映射到其对应的嵌入向量
            author2embedding = {}
            
            
            
           


            with open(graph_embedding_file, encoding='utf-8') as fp:  # smart open can read .gz files
                for i, line in enumerate(fp):
                    cols = line.split('\t')

                    entity_id = cols[0]
                    
                    # 这种格式用于识别Wikidata项目的QID（Wikidata标识符）
                    if entity_id.startswith('<http://www.wikidata.org/entity/Q') and entity_id.endswith('>'):
                        entity_id = entity_id.replace('<http://www.wikidata.org/entity/', '').replace('>', '')

                        if entity_id in selected_entity_ids:
                            author2embedding[qid2author[entity_id]] = np.array(cols[1:]).astype(np.float64)

                    if not i % 100000:
                        print(f'Lines completed {i}')

            # 保存作者到嵌入向量的映射
            with open(out_file, 'wb') as f:
                pickle.dump(author2embedding, f)
            
            print(f'Saved to {out_file}')
            
            
# 将嵌入向量转换成 Tensorflow Projector 的格式 (可选)
def convert_for_projector(author2embedding_path, out_projector_path, out_projector_meta_path):
    """

    Converts embeddings such that they can be visualized with Tensorflow Projector

    See http://projector.tensorflow.org/

    :param author2embedding_path: Path to output of `run()`
    :param out_projector_path: Write TSV file of vectors to this path.
    :param out_projector_meta_path: Write TSV file of metadata to this path.
    
    
    初始化两个列表vecs和metas，分别用于存储向量和对应的元数据（即作者名称）。
    遍历字典中的每一项，对于每一个作者及其对应的嵌入向量执行以下操作：
    
        将向量转换为字符串形式，并用制表符\t分隔每个元素，然后添加到vecs列表中。
        将作者名称添加到metas列表中
    """
    print(f'Reading embeddings from {author2embedding_path}')

    with open(author2embedding_path, 'rb') as f:
        a2vec = pickle.load(f)

        vecs = []
        metas = []

        for a, vec in a2vec.items():
            vecs.append('\t'.join([str(t) for t in vec.tolist()]))
            metas.append(a)

        with open(out_projector_path, 'w') as ff:
            ff.write('\n'.join(vecs))

        with open(out_projector_meta_path, 'w') as ff:
            ff.write('\n'.join(metas))


if __name__ == '__main__':
    fire.Fire()