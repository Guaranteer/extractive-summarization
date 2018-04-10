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
        self.model_path = os.path.join(self.data_params['cache_dir'], 'tfmodel')

        self.model.build_graph()
        self.model_saver = tf.train.Saver()

        self.train_dataset = dg.Batcher(params,'train')
        self.valid_dataset = dg.Batcher(params, 'valid')
        self.test_dataset = dg.Batcher(params, 'test')


    def train(self):
        print('Training begins .............. ')
        global_step = tf.get_variable('global_step',[],initializer=tf.constant_initializer(0),trainable=False)
        learn_rates = tf.train.exponential_decay(self.params['learning_rate'],global_step,
                                                 decay_rate=self.params['decay_rate'],decay_steps=self.params['decay_steps'])
        optimizer = tf.train.AdamOptimizer(learning_rate=learn_rates)
        train_op = optimizer.minimize(self.model.loss, global_step= global_step)

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
                if i_batch % self.params[10] == 0:
                    t2 = time.time()
                    print('Epoch %d, Batch %d, loss = %.4f, %.3f seconds/batch' % (i_epoch, i_batch, loss, (t2 - t1) / 10))
                    t1 = t2

            avg_batch_loss = sum(loss_sum)/len(loss_sum)
            t_end = time.time()
            if i_epoch % 5 == 0:
                print ('********************************************************')
                print ('Overall evaluation')
                valid_acc, _ = self._test(self.sess,i_epoch)
                print ('Epoch %d ends. Average loss %.3f. %.3f seconds/epoch' % (i_epoch, avg_batch_loss, t_end-t_begin))
                print ('********************************************************')
            else:
                print ('********************************************************')
                print ('Epoch %d ends. Average loss %.3f. %.3f seconds/epoch' % (i_epoch, avg_batch_loss, t_end-t_begin))
                valid_acc = self._evaluate()
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
        type_count = np.zeros(self.data_params['n_types'], dtype=float)
        correct_count = np.zeros(self.data_params['n_types'], dtype=float)
        wups_count = np.zeros(self.data_params['n_types'], dtype=float)
        wups_count2 = np.zeros(self.data_params['n_types'], dtype=float)

        num_per_epoch = len(batcher.keys) // 100
        for _ in range(num_per_epoch):
            img_frame_vecs, img_frame_n, ques_vecs, ques_n, ans_vec, type_vec = batcher.generate()
            if ans_vec is None:
                break
            batch_data = {
                model.input_q: ques_vecs,
                model.y: ans_vec
            }

            batch_data[model.input_x] = img_frame_vecs
            batch_data[model.input_x_len] = img_frame_n
            batch_data[model.input_q_len] = ques_n
            batch_data[model.is_training] = False

            predict_status, top_index = \
                sess.run([model.predict_status, model.top_index], feed_dict=batch_data)

            for i in range(len(type_vec)):
                type_count[type_vec[i]] += 1
                correct_count[type_vec[i]] += predict_status[i]

                ground_a = batcher.ans_set[ans_vec[i]]
                generate_a = batcher.ans_set[top_index[i][0]]

                wups_value = wups.compute_wups([ground_a], [generate_a], 0.7)
                wups_value2 = wups.compute_wups([ground_a], [generate_a], 0.9)
                wups_count[type_vec[i]] += wups_value
                wups_count2[type_vec[i]] += wups_value2

        print('****************************')
        acc = correct_count.sum() / type_count.sum()
        wup_acc = wups_count.sum() / type_count.sum()
        wup_acc2 = wups_count2.sum() / type_count.sum()
        print('Overall Accuracy (top 1):', acc, '[', correct_count.sum(), '/', type_count.sum(), ']')
        print('Overall Wup (@0):', wup_acc, '[', wups_count.sum(), '/', type_count.sum(), ']')
        print('Overall Wup (@0.9):', wup_acc2, '[', wups_count2.sum(), '/', type_count.sum(), ']')
        type_acc = [correct_count[i] / type_count[i] for i in range(self.data_params['n_types'])]
        type_wup_acc = [wups_count[i] / type_count[i] for i in range(self.data_params['n_types'])]
        type_wup_acc2 = [wups_count2[i] / type_count[i] for i in range(self.data_params['n_types'])]
        print('Accuracy for each type:', type_acc)
        print('Wup@0 for each type:', type_wup_acc)
        print('Wup@0.9 for each type:', type_wup_acc2)
        print(type_count)

        return acc