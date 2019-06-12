import pandas as pd
from pathlib import Path
import numpy as np
import math
import pickle 

def embedding_item_diction(df_meta):
    items = df_meta['item_id'].values
    properties = df_meta['properties'].values
    all_items_test = []
    for i in range(len(properties)):
        p_data = properties[i].split('|')
        all_items_test += p_data
    set_items = set(all_items_test)

    list_items = list(set_items)
    list_items.sort()
    all_items = {}
    for i in range(len(properties)):
        all_items[items[i]] = properties[i].split('|')
    embedding_items = {}
    #print(len(list_items))
    for keys in all_items.keys():
        item_embeddings = [0]*len(list_items)
        for i in range(len(list_items)):
            if list_items[i] in all_items[keys]:
                item_embeddings[i] = 1
        embedding_items[keys] = item_embeddings
    return embedding_items
    
def make_data(data): 
    data_split = []
    temp_data = []
    train_size = len(data)
    action_index = 1
    for i in range(train_size):
        if(i==0):
            temp_data.append(data[i])
        if(i>0):
            if(data[i,action_index] == data[i-1,action_index]):
                temp_data.append(data[i])
            else:
                data_split.append(np.array(temp_data))
                temp_data = []
                temp_data.append(data[i])
    return data_split
    
def data_preprocess(data): 
    new_data = []
    for i in range(len(data)):
        check_data = False
        final_click = 0
        for j in range(len(data[i])):
            if(data[i][j,4] =='clickout item'):
                final_click = j
                check_data = True
        if(final_click < len(data[i])-1 and check_data):
            new_data.append(np.delete(data[i],np.s_[final_click+1:],0))
        elif(check_data):
            new_data.append(data[i])
    return new_data


              

class Data:
    def __init__(self):
        path = Path('./')
        train_path = path.joinpath('train.csv')
        test_path = path.joinpath('test.csv')
        submission_path = path.joinpath('submission_popular.csv')
        metadata_path = path.joinpath('item_metadata.csv')

        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        df_meta = pd.read_csv(metadata_path)
        df_submission = pd.read_csv(submission_path)

        action_list = set(df_train['action_type'].values)
        reference_list = set(df_train['reference'].values)
        platform_list = set(df_train['platform'].values)
        city_list = set(df_train['city'].values)
        device_list = set(df_train['device'].values)
        filter_list = set(df_train['current_filters'].values)
        filter_list =  [x for x in filter_list if str(x) != 'nan']

        action_list = list(action_list)
        action_list.sort()
        self.action_list = action_list
        
        reference_list = list(reference_list)
        reference_list.sort()
        self.reference_list = reference_list
        
        platform_list = list(platform_list)
        platform_list.sort()
        self.platform_list = platform_list
        
        city_list = list(city_list)
        city_list.sort()
        self.city_list = city_list
        
        
        device_list = list(device_list)
        device_list.sort()
        self.device_list = device_list
        
        filter_list = list(filter_list)
        filter_list.sort()
        self.filter_list = filter_list
        
        self.reference_diction = embedding_item_diction(df_meta)
        self.train_data_preprocess = data_preprocess(make_data(df_train.values))
        self.train_data_preprocess = self.check_inference(self.train_data_preprocess)
        print(len(self.train_data_preprocess))
        #self.MakeEmbeddingVector_task1(self.train_data_preprocess[0])
        
    def check_inference(self, data):
        datalength = len(data)
        maxlength = 0
        correct_data = 0
        new_data = []
        for i in range(datalength):
            if(data[i][-1,5] in data[i][-1,10].split('|')):
                if int(data[i][-1,5]) in self.reference_diction:
                    new_data.append(data[i])
                    correct_data += 1
            else:
                pass
        return new_data    

    def RepresentsInt(self, s):
        try: 
            int(s)
            return True
        except ValueError:
            return False     
        
    def sum_list(self, list1, list2):
        length = len(list1)
        final_list = []
        for i in range(length):
            final_list.append(list1[i] +list2[i])
        return final_list 
    #filter embedding 보조함수
    
    def filter_embedding(self, data, version):
        all_filters = []
        for i in range(len(self.filter_list)):
            if i>0:
                all_filters += self.filter_list[i].split('|')
        all_filter_set = set(all_filters)
        all_filter_list = list(all_filter_set)
        all_filter_list.sort()
        embedding_data = [0]*202
        if version == 1:
            datalist = data.split('|')
            for filters in datalist:
                embedding_data[all_filter_list.index(filters)] = 1
            return embedding_data
    def one_hot_encoding(self, number, length):
        return [int(i==number) for i in range(length)]
    
    def embedding(self, data, action_index):
        if action_index == 0: #user_id dim 0
            return []

        if action_index == 1: # session_id dim 0
            return []

        if action_index == 2: # timestep dim 0
            return []

        if action_index == 3: #step dim 1
            return []

        if action_index == 4: #action_type dim 10
            return self.one_hot_encoding(self.action_list.index(data), 10)

        if action_index == 5: #reference dim 157
            if self.RepresentsInt(data):
                return self.reference_diction[int(data)]
            else:
                return [0]*157

        if action_index == 6: #platform dim 55
            return self.one_hot_encoding(self.platform_list.index(data), 55)

        if action_index == 7: #city dim 0
            #return city_list.index(data)
            return []

        if action_index == 8: #device dim 3
            return self.one_hot_encoding(self.device_list.index(data), 3)

        if action_index == 9: #current_filters dim 202
            if(isinstance(data, float) and math.isnan(data) == True):
                return [0]*202
            else:
                return self.filter_embedding(data, 1)

        if action_index == 10: #impressions dim 157
            if(isinstance(data, float) and math.isnan(data) == True):
                return [0]*157
            else:
                impression_embedding = [0]*157
                impression_list = data.split('|')
                for impression in impression_list:
                    try:
                        impression_embedding = self.sum_list(impression_embedding, self.reference_diction[int(impression)])

                    except:
                        pass
                return impression_embedding


        if action_index == 11: #prices dim 25
            if(isinstance(data, float) and math.isnan(data) == True):
                return [0]*25
            else:
                price_embedding = []
                for prices in data.split('|'):
                    price_embedding.append(int(prices))
                if(len(price_embedding) < 25):
                    price_embedding += [0]*(25-len(price_embedding))   
                return price_embedding    
            
    def MakeEmbeddingVector_task1(self,data):
        item_vector = [] # item data embedding
        num_action = len(data)
        correct_index = []
        reference_vector = []
        filter_vector = [0]*202
        current_reference  = ''
        reference_size = 0
        for i in range(num_action-1, -1, -1): #거꾸로
            timestep_vector = []
            if i != num_action-1 and self.RepresentsInt(data[i, 5]) and current_reference != data[i,5] and reference_size < 2:
                current_reference = data[1,5]
                if int(data[i, 5]) in self.reference_diction:
                    reference_size += 1
                    reference_vector += self.reference_diction[int(data[i, 5])]
            if(filter_vector == [0]*202):
                filter_vector = self.embedding(data[i,9], 9)
        reference_vector += [0]*(314-len(reference_vector))
        session_vector = self.embedding(data[-1, 4], 4) + reference_vector + self.embedding(data[-1, 6], 6) + self.embedding(data[-1, 8], 8) + filter_vector
        # 10+314+202+55+3 = 584

        temp_impressions = data[-1, 10].split('|')
        for i in range(len(temp_impressions)):
            if(data[-1,5] == temp_impressions[i]):
                correct_index.append(i)
            if int(temp_impressions[i]) in self.reference_diction:
                item_vector.append(self.reference_diction[int(temp_impressions[i])])
            else:
                item_vector.append([0]*157)
        return session_vector, item_vector, correct_index
    
    def get_MRS(self, score, idx):
        return (list(np.squeeze(np.argsort(score, axis = 0))).index(idx)+1)/np.shape(score)[0]
