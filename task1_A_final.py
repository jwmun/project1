import data_preprocess
import numpy as np
import tensorflow as tf

class TaskA:
    def __init__(self):
        self.data = data_preprocess.Data()

        self.train_data_preprocess = self.data.train_data_preprocess

        self.data_num = len(self.train_data_preprocess)
        self.train_embedding_dim= 584
        self.item_embedding_dim = 157
        self.latent_dim = 200
        self.learning_rate = 1e-3

        self.session_feature = tf.get_variable("session_feature", [self.train_embedding_dim, self.latent_dim], initializer=tf.random_normal_initializer())
        self.item_feature = tf.get_variable("item_feature", [self.item_embedding_dim, self.latent_dim], initializer=tf.random_normal_initializer())
        self.session_bias = tf.get_variable("session_bias", [self.train_embedding_dim], initializer=tf.random_normal_initializer())
        self.item_bias = tf.get_variable("item_bias", [self.item_embedding_dim], initializer=tf.random_normal_initializer())

        self.train_data = tf.placeholder(tf.float32, shape=(self.train_embedding_dim))
        self.train_items  = tf.placeholder(tf.float32, shape=(None, self.item_embedding_dim))
        self.train_index = tf.placeholder(tf.int32, [1])

        self.scores = tf.matmul(tf.matmul(self.train_items, self.item_feature), tf.transpose(tf.matmul(tf.expand_dims(self.train_data,0), self.session_feature))) + tf.matmul(tf.expand_dims(self.train_data,0), tf.expand_dims(self.session_bias,1)) + tf.matmul(self.train_items, tf.expand_dims(self.item_bias,1))

        self.loss = -tf.reduce_mean(tf.math.log(tf.sigmoid(self.scores[self.train_index[0]] - self.scores)))
        self.loss2 = tf.reduce_mean(tf.sigmoid(self.scores- self.scores[self.train_index[0]]) + tf.sigmoid(tf.math.pow(self.scores,2)))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss2)
        self.saver = tf.train.Saver()
        
        self.final_train_data = self.train_data_preprocess[:810000]
        self.final_valid_data = self.train_data_preprocess[810000:811000]
        self.final_test_data = self.train_data_preprocess[811000:]
        
        self.num_train = len(self.final_train_data)
        self.num_valid = len(self.final_valid_data)
        self.num_test = len(self.final_test_data)
        
    def train(self):
        max_mrs = 0
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            mean_loss = 0
            mean_rs = 0
            data_number = 0
            for i in range(self.num_train):

                batch_data, batch_items, correct_index = self.data.MakeEmbeddingVector_task1(self.final_train_data[i])
                batch_data  = np.asarray(batch_data)
                batch_items = np.asarray(batch_items)
                #print(correct_index)
                sess.run(self.optimizer, feed_dict = {self.train_data : batch_data, self.train_items : batch_items, self.train_index : correct_index})
                ls = sess.run(self.loss2, feed_dict = {self.train_data : batch_data, self.train_items : batch_items, self.train_index : correct_index})
                _score = sess.run(self.scores, feed_dict = {self.train_data : batch_data, self.train_items : batch_items, self.train_index : correct_index})
                
                if(len(_score) == 1):
                    current_rs = 1
                else:
                    current_rs = self.data.get_MRS(_score, correct_index[0])

                mean_rs += current_rs 
                mean_loss += ls
                data_number +=1
                if (i % 1000 == 1):
                    valid_mean_rs = 0
                    valid_mean_loss = 0
                    for j in range(self.num_valid):
                        valid_batch_data, valid_batch_items, valid_correct_index = self.data.MakeEmbeddingVector_task1(self.final_valid_data[j])
                        valid_batch_data  = np.asarray(valid_batch_data)
                        valid_batch_items = np.asarray(valid_batch_items)
                        valid_loss = sess.run(self.loss2, feed_dict = {self.train_data : valid_batch_data, self.train_items : valid_batch_items, self.train_index : valid_correct_index})
                        valid_score = sess.run(self.scores, feed_dict = {self.train_data : valid_batch_data, self.train_items : valid_batch_items, self.train_index : valid_correct_index})
                        if(len(valid_score) == 1):
                            valid_current_rs = 1
                        else:
                            valid_current_rs = self.data.get_MRS(valid_score, valid_correct_index[0])
                        valid_mean_rs += valid_current_rs
                        valid_mean_loss += valid_loss

                    if(valid_mean_rs / self.num_valid > max_mrs):
                        max_mrs = valid_mean_rs / self.num_valid
                        print('session saved')
                        self.saver.save(sess, './saved_model/task1_ver3')

                    print('step', i)
                    print("output index is : ", np.argmax(_score, axis = 0)[0])
                    print("correct index is : ", correct_index[0])
                    print("corrent rs is : ", current_rs)

                    print("train mean loss is : ", mean_loss / data_number)
                    print("valid mean loss is : ", valid_mean_loss / self.num_valid)

                    print("train mean reciprocal score is : ", mean_rs / data_number)
                    print("valid mean reciprocal score is : ", valid_mean_rs / self.num_valid)

                    print("data num is : ", data_number)

                    mean_rs =0
                    mean_loss = 0
                    data_number = 0


    def test(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            #mean_loss = 0
            mean_rs = 0
            self.saver.restore(sess, './saved_model/task1_ver2')
            num_test = len(self.final_test_data)
            for i in range(num_test):

                    batch_data, batch_items, correct_index = self.data.MakeEmbeddingVector_task1(self.final_test_data[i])
                    batch_data  = np.asarray(batch_data)
                    batch_items = np.asarray(batch_items)

                    #sess.run(optimizer, feed_dict = {train_data : batch_data, train_items : batch_items, train_index : correct_index})
                    #ls = sess.run(loss2, feed_dict = {train_data : batch_data, train_items : batch_items, train_index : correct_index})
                    _score = sess.run(self.scores, feed_dict = {self.train_data : batch_data, self.train_items : batch_items})
                    if(len(_score) == 1):
                        current_rs = 1
                    else:
                        current_rs = self.data.get_MRS(_score, correct_index[0])

                    mean_rs += current_rs 
                    #mean_loss += ls

            #print("mean loss is : ", mean_loss /num_test)
            print("mean reciprocal score is : ", mean_rs /num_test)
            return mean_rs / num_test           

