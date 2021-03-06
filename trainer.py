import tensorflow as tf
import numpy as np
import os
import time
import data_generator as dg

class Trainer(object):

    def __init__(self,params,model):
        self.model = model
        self.params = params

        sess_config =  tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = sess_config)
        self.model_path = os.path.join(self.params['cache_dir'], 'tfmodel')

        self.model.build_graph()
        self.model_saver = tf.train.Saver()

        print('Load train dataset')
        self.train_dataset = dg.Batcher(params,'train')
        print('Load valid dataset')
        self.valid_dataset = dg.Batcher(params, 'valid')
        print('Load test dataset')
        self.test_dataset = dg.Batcher(params, 'test')


    def train(self):
        print('Training begins .............. ')
        global_step = tf.get_variable('global_step',[],initializer=tf.constant_initializer(0),trainable=False)
        learn_rates = tf.train.exponential_decay(self.params['learning_rate'],global_step,
                                                 decay_rate=self.params['decay_rate'],decay_steps=self.params['decay_steps'])
        optimizer = tf.train.AdamOptimizer(learning_rate=learn_rates)
        train_op = optimizer.minimize(self.model.loss, global_step= global_step)


        if not os.path.exists(self.model_path):
            print('create path: ', self.model_path)
            os.makedirs(self.model_path)

        init  = tf.global_variables_initializer()
        self.sess.run(init)
        best_epoch_acc = 0
        best_epoch_id = 0

        print('****************************')
        print('Trainning datetime:', time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        print('params')
        print(self.params)
        print('****************************')

        for  i_epoch in range(self.params['max_epoches']):
            t_begin = time.time()
            t1 = time.time()
            self.train_dataset.reset()
            loss_sum = list()
            num_per_epoch = self.train_dataset.num

            for i_batch in range(num_per_epoch):
                org_docs, query_vec, q_len, doc_vec, d_len, rewards = self.train_dataset.get_data()

                feed = {self.model.input_q: query_vec, self.model.input_q_len: q_len, self.model.input_s:doc_vec, self.model.input_s_len:d_len, self.model.y:rewards}

                _, loss = self.sess.run([train_op, self.model.loss], feed_dict = feed)


                # length = len(rewards)
                # re_list = list()
                # for _ in range(length):
                #     feed = {self.model.input_q: query_vec, self.model.input_q_len: q_len,
                #             self.model.input_s: doc_vec, self.model.input_s_len: d_len,
                #             self.model.y: rewards, self.model.last_x: last_x,
                #             self.model.input_h: h_now}
                #     _, loss, idx, h_now, last_x, re, soft = self.sess.run([train_op, self.model.loss, self.model.chose,
                #                     self.model.h_now, self.model.chose_s, self.model.chose_r, self.model.softmax], feed_dict = feed)
                #     print(loss)
                #     print(re)
                #     print(soft)
                #     print(rewards)
                #     re_list.append(re)
                #     last_x = last_x.reshape([self.model.L,1])
                #     doc_vec = np.delete(doc_vec, idx, axis=0)
                #     d_len = np.delete(d_len, idx, axis=0)
                #     rewards = np.delete(rewards, idx, axis=0)
                #
                #     loss_sum.append(loss)

                # values = self.dataset.discount_rewards(re_list,0.80)

                loss_sum.append(loss)
                if i_batch % self.params['display_batch_interval'] == 0:
                    t2 = time.time()
                    print('Epoch %d, Batch %d, loss = %.4f, %.3f seconds/batch' % (i_epoch, i_batch, sum(loss_sum)/len(loss_sum), (t2 - t1) / 10))
                    t1 = t2

            avg_batch_loss = sum(loss_sum)/len(loss_sum)
            t_end = time.time()
            if i_epoch % 5 == 0:
                print ('********************************************************')
                print ('Overall evaluation')
                valid_acc, _ = self._test(i_epoch)
                print ('Epoch %d ends. Average loss %.3f. %.3f seconds/epoch' % (i_epoch, avg_batch_loss, t_end-t_begin))
                print ('********************************************************')
            else:
                print ('********************************************************')
                print ('Epoch %d ends. Average loss %.3f. %.3f seconds/epoch' % (i_epoch, avg_batch_loss, t_end-t_begin))
                valid_acc = self._evaluate(self.valid_dataset)
                print ('********************************************************')

            if valid_acc > best_epoch_acc:
                best_epoch_acc = valid_acc
                best_epoch_id = i_epoch
                print ('Saving new best model...')
                timestamp = time.strftime("%m%d%H%M%S", time.localtime())
                self.last_checkpoint = self.model_saver.save(self.sess, self.model_path+timestamp, global_step=global_step)
                print ('Saved at', self.last_checkpoint)
            else:
                if i_epoch-best_epoch_id >= self.params['early_stopping']:
                    print ('Early stopped. Best loss %.3f at epoch %d' % (best_epoch_acc, best_epoch_id))
                    break

    def _evaluate(self, batcher):
        # evaluate the model in a set
        batcher.reset()

        num_per_epoch = batcher.num

        scores = list()
        for i_batch in range(num_per_epoch):
            org_docs, query_vec, q_len, doc_vec, d_len, rewards = batcher.get_data()

            feed = {self.model.input_q: query_vec, self.model.input_q_len: q_len, self.model.input_s:doc_vec, self.model.input_s_len:d_len, self.model.y:rewards}

            loss, prob = self.sess.run([self.model.loss, self.model.prob], feed_dict = feed)
            index = list(np.argsort(prob))
            index.reverse()

            score_list = [rewards[idx] for idx in index[0:10]]
            score = sum(score_list)
            scores.append(score)

        score_mean = sum(scores)/len(scores)
        print('****************************')
        print ('Overall Scores :', scores)
        print('Mean Score :', score_mean)

        return score_mean

    def _test(self, i_epoch):
        print('Validation set:')
        valid_acc = self._evaluate(self.valid_dataset)
        print('Test set:')
        test_acc = self._evaluate(self.test_dataset)
        return valid_acc, test_acc

