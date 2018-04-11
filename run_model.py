from model import Model
from trainer import Trainer

params = {
    'input_word_dim': 300,
    's_max_len': 50,
    'q_max_len': 50,
    'depth': 40,
    'L': 3000,
    'K': 1500,
    'regularization_beta': 1e-7,
    'discount': 0.99,

    'querySource_train': './data/duc2006/duc2006_topics.sgml',
    'querySource_valid': './data/duc2005/duc2005_topics.sgml',
    'querySource_test': './data/duc2007/duc2007_topics.sgml',
    'pred_csv_train': './data/duc2006/pred.csv',
    'pred_csv_valid': './data/duc2005/pred.csv',
    'pred_csv_test': './data/duc2007/pred.csv',
    'name_len_train': 5,
    'name_len_valid': 4,
    'name_len_test': 5,

    'learning_rate': 1e-3,
    'decay_steps': 3000,
    'decay_rate': 0.8,
    'max_epoches': 100,
    'early_stopping': 5,
    'display_batch_interval':10,
    'cache_dir':'./save_model/',
    'word2vec':'./word2vec/word2vec.bin'

}


if __name__ == '__main__':
    model = Model(params)
    trainer = Trainer(params, model)
    trainer.train()