import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

import keras
import keras.backend as K
from keras.models import Model, load_model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy
from keras.layers import Input, Dense, Dropout, CuDNNGRU, Embedding, Lambda, Reshape
from keras import layers


class SessionDataset:
    """Credit to yhs-968/pyGRU4REC."""    
    def __init__(self, data, sep='\t', session_key='SessionId', item_key='ItemId', time_key='Time', n_samples=-1, itemmap=None, time_sort=False):
        """
        Args:
            path: path of the csv file
            sep: separator for the csv
            session_key, item_key, time_key: name of the fields corresponding to the sessions, items, time
            n_samples: the number of samples to use. If -1, use the whole dataset.
            itemmap: mapping between item IDs and item indices
            time_sort: whether to sort the sessions by time or not
        """
        self.df = data
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.time_sort = time_sort
        self.add_item_indices(itemmap=itemmap)
        self.df.sort_values([session_key, time_key], inplace=True)

        # Sort the df by time, and then by session ID. That is, df is sorted by session ID and
        # clicks within a session are next to each other, where the clicks within a session are time-ordered.

        self.click_offsets = self.get_click_offsets()
        self.session_idx_arr = self.order_session_idx()
        
    def get_click_offsets(self):
        """
        Return the offsets of the beginning clicks of each session IDs,
        where the offset is calculated against the first click of the first session ID.
        """
        offsets = np.zeros(self.df[self.session_key].nunique() + 1, dtype=np.int32)
        # group & sort the df by session_key and get the offset values
        offsets[1:] = self.df.groupby(self.session_key).size().cumsum()

        return offsets

    def order_session_idx(self):
        """ Order the session indices """
        if self.time_sort:
            # starting time for each sessions, sorted by session IDs
            sessions_start_time = self.df.groupby(self.session_key)[self.time_key].min().values
            # order the session indices by session starting times
            session_idx_arr = np.argsort(sessions_start_time)
        else:
            session_idx_arr = np.arange(self.df[self.session_key].nunique())

        return session_idx_arr
    
    def add_item_indices(self, itemmap=None):
        """ 
        Add item index column named "item_idx" to the df
        Args:
            itemmap (pd.DataFrame): mapping between the item Ids and indices
        """
        if itemmap is None:
            item_ids = self.df[self.item_key].unique()  # unique item ids
            print('------------------------------------------------------------------------------\n')
#             print('len(item_ids)\n')
#             print(len(item_ids))
#             print('\n')
            item2idx = pd.Series(data=np.arange(len(item_ids)),  #7621(350) 7310(540)7305(5)7589(2) 7622 1890 +7496 1(7621 )4(2217) 7(1206) #0(28320) #7621(9,360) model=>7622(1) +1406是為了讓id跟前面的 1 資料集不重複
                                 index=item_ids)
            itemmap = pd.DataFrame({self.item_key:item_ids,
                                   'item_idx':item2idx[item_ids].values})  # 在此加上另一個類別資料的item len讓id不重複
        
        self.itemmap = itemmap
        self.df = pd.merge(self.df, self.itemmap, on=self.item_key, how='inner')
        
    @property    
    def items(self):
        return self.itemmap.ItemId.unique()
        

class SessionDataLoader:
    """Credit to yhs-968/pyGRU4REC."""    
    def __init__(self, dataset, batch_size=50):
        """
        A class for creating session-parallel mini-batches.
        Args:
            dataset (SessionDataset): the session dataset to generate the batches from
            batch_size (int): size of the batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.done_sessions_counter = 0
        
    def __iter__(self):
        """ Returns the iterator for producing session-parallel training mini-batches.
        Yields:
            input (B,):  Item indices that will be encoded as one-hot vectors later.
            target (B,): a Variable that stores the target item indices
            masks: Numpy array indicating the positions of the sessions to be terminated
        """

        df = self.dataset.df
        session_key='SessionId'
        item_key='ItemId'
        time_key='TimeStamp'
        self.n_items = df[item_key].nunique()+1
        click_offsets = self.dataset.click_offsets
        session_idx_arr = self.dataset.session_idx_arr

        iters = np.arange(self.batch_size)
        print('iters:')
        print(iters)
        print('\n')
        maxiter = iters.max()
        print('length of session_idx_arr')
        print('\n')
        print(len(session_idx_arr))
        print('\n')
        start = click_offsets[session_idx_arr[iters]]
#         print('length of start:')
#         print(len(start))
        
        #print('start:')
        #print(start)
        #print('\n')
        end = click_offsets[session_idx_arr[iters] + 1]
        mask = [] # indicator for the sessions to be terminated
        finished = False        

        while not finished:
            minlen = (end - start).min()
            # Item indices (for embedding) for clicks where the first sessions start
            idx_target = df.item_idx.values[start]
            for i in range(minlen - 1):
                # Build inputs & targets
                idx_input = idx_target
                idx_target = df.item_idx.values[start + i + 1]
                inp = idx_input
                target = idx_target
                yield inp, target, mask
                
            # click indices where a particular session meets second-to-last element
            start = start + (minlen - 1)
            # see if how many sessions should terminate
            mask = np.arange(len(iters))[(end - start) <= 1]
            self.done_sessions_counter = len(mask)
            for idx in mask:
                maxiter += 1
                if maxiter >= len(click_offsets) - 1:
                    finished = True
                    break
                # update the next starting/ending point
                iters[idx] = maxiter
                start[idx] = click_offsets[session_idx_arr[maxiter]]
                end[idx] = click_offsets[session_idx_arr[maxiter] + 1]


def create_model(args):   
    emb_size = 50
    hidden_units = 100
    size = emb_size

    inputs = Input(batch_shape=(args.batch_size, 1, args.train_n_items), name='origin_input') # 
    gru, gru_states = CuDNNGRU(hidden_units, stateful=True, return_state=True, name='origin_gru')(inputs)#return_sequences
#     gru2, gru_states2 = CuDNNGRU(hidden_units, stateful=True, return_sequences=True, name='origin_gru2')(gru) # test 2 layer gru
    drop2 = Dropout(0.25)(gru)
    predictions = Dense(args.train_n_items, activation='softmax', name='new_softmax')(drop2) #args.train_n_items
    model = Model(input=inputs, output=[predictions])
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=categorical_crossentropy, optimizer=opt)
    model.summary()

    filepath='./model_checkpoint.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=2, save_best_only=True, mode='min')
    callbacks_list = []
    return model


def create_vae_model(args):
    emb_size = 50
    hidden_units = 100
    size = emb_size
    
    # 抽掉input
    no = load_model('noInput_vae_12240.h5')
    
    inputs = Input(batch_shape=(args.batch_size, 1, args.train_n_items),name='origin_input')
    inputs2 = no(inputs)
    h1 = Reshape((1, args.train_n_items))(inputs2)
    gru, gru_states = CuDNNGRU(hidden_units, stateful=True, return_state=True, name='origin_gru')(h1)
    drop2 = Dropout(0.25,name='origin_dropout')(gru)
    predictions = Dense(args.train_n_items, activation='softmax', name='origin_softmax')(drop2) # 做transfer learning的時候,記得將name從origin改成new 不要讓新model的softmax層載入舊weights
    model = Model(input=inputs, output=[predictions])
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=categorical_crossentropy, optimizer=opt)
    model.summary()
    
    filepath='./model_checkpoint.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=2, save_best_only=True, mode='min')
    callbacks_list = []
    return model


def get_states(model):
    return [K.get_value(s) for s,_ in model.state_updates]


def get_metrics(model, args, train_generator_map, recall_k=20, mrr_k=20):

    test_dataset = SessionDataset(args.test_data, itemmap=train_generator_map)
    test_generator = SessionDataLoader(test_dataset, batch_size=args.batch_size)

    n = 0
    rec_sum = 0
    mrr_sum = 0

    with tqdm(total=args.test_samples_qty) as pbar:
        for feat, label, mask in test_generator:
#             print('line 231',label)
#             print()
            target_oh = to_categorical(label, num_classes=args.train_n_items) # args.train_n_items 37098
            input_oh  = to_categorical(feat,  num_classes=args.train_n_items) # args.train_n_items
            input_oh = np.expand_dims(input_oh, axis=1)
            
            pred = model.predict(input_oh, batch_size=args.batch_size)
            
            for row_idx in range(feat.shape[0]):
                pred_row = pred[row_idx] 
                label_row = target_oh[row_idx]

                rec_idx =  pred_row.argsort()[-recall_k:][::-1]
                mrr_idx =  pred_row.argsort()[-mrr_k:][::-1]
                tru_idx = label_row.argsort()[-1:][::-1]

                n += 1

                if tru_idx[0] in rec_idx:
                    rec_sum += 1

                if tru_idx[0] in mrr_idx:
                    mrr_sum += 1/int((np.where(mrr_idx == tru_idx[0])[0]+1))
            
            pbar.set_description("Evaluating model")
            pbar.update(test_generator.done_sessions_counter)
#     print('n:',n)
#     print('predict succese:',rec_sum)
    recall = rec_sum/n
    mrr = mrr_sum/n
    return (recall, recall_k), (mrr, mrr_k)


def train_model(model, args, save_weights = False):
    train_dataset = SessionDataset(args.train_data)
    model_to_train = model
#     model_to_train.layers[1].trainable = False
    batch_size = args.batch_size
#     print('train...')
    for epoch in range(1, 10):
        with tqdm(total=args.train_samples_qty) as pbar:
            loader = SessionDataLoader(train_dataset, batch_size=batch_size)
            for feat, target, mask in loader:
                
                real_mask = np.ones((batch_size, 1))
                for elt in mask:
                    real_mask[elt, :] = 0

                hidden_states = get_states(model_to_train)[0]
                hidden_states = np.multiply(real_mask, hidden_states)
                hidden_states = np.array(hidden_states, dtype=np.float32)
                model_to_train.layers[1].reset_states(hidden_states) # gru 那層(如果要預訓練就代3微調代1)
#                 model_to_train.layers[2].reset_states(hidden_states)
#                 print('feat:',len(feat))
                input_oh = to_categorical(feat, num_classes=args.train_n_items) # loader.n_items
                input_oh = np.expand_dims(input_oh, axis=1)

                target_oh = to_categorical(target, num_classes=args.train_n_items) # loader.n_items

                tr_loss = model_to_train.train_on_batch(input_oh, target_oh)

                pbar.set_description("Epoch {0}. Loss: {1:.5f}".format(epoch, tr_loss))
                pbar.update(loader.done_sessions_counter)
            
        if save_weights:
            print("Saving weights...")
            model_to_train.save('./GRU4REC_1_{}.h5'.format(epoch))
        
        (rec, rec_k), (mrr, mrr_k) = get_metrics(model_to_train, args, train_dataset.itemmap)

        print("\t - Recall@{} epoch {}: {:5f}".format(rec_k, epoch, rec))
        print("\t - MRR@{}    epoch {}: {:5f}".format(mrr_k, epoch, mrr))
        print("\n")

        

def transfer_learning(model, args, save_weights = False):
    '''
    把訓練好的model選擇需要的權重留存,然後再用target domain data 來 tune
    或者把輸出層用 target domain data 訓練後接上,之前訓練好的model
    '''
    # 拿出之前訓練好的vae模型去掉vae後再來拿小資料作微調
    
    train_dataset = SessionDataset(args.train_data)
    model_to_train = model # temp_model if vae type
    batch_size = args.batch_size 
    model_to_train.load_weights('GRU4REC_1_9.h5', by_name=True)  # 原本的
# vae type    
#     temp_model.load_weights('GRU4REC_1_vae_9.h5', by_name=True) #將 0 訓練出來的model的weights載入
    
#     m0 = temp_model.layers[0]
#     m1 = temp_model.layers[1]
#     m2 = temp_model.layers[2]
#     m3 = temp_model.layers[3]
#     m4 = temp_model.layers[4]
#     m5 = temp_model.layers[5]
    
#     y = m0
#     y, yy = m3(y.get_output_at(0))
#     y = m4(y)
#     y = m5(y)
    
#     model_to_train = Model(input=m0.input, output=[y])
#     opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#     model_to_train.compile(loss=categorical_crossentropy, optimizer=opt)
#     model_to_train.summary()
    
    for layer in model_to_train.layers:   #將每一層設成不能訓練
        layer.trainable = False
    
#     model_to_train.layers[2].trainable = True
    model_to_train.layers[-1].trainable = True  #將最後一層softmax層設成可訓練,因為要讓 0 的model只用S的DATA訓練最後一層,再來比較前後差別
    #model_to_train.layers[0].trainable = True  # 只將最後一層做調整
    print('\nset model done\n')
    for epoch in range(1, 10):
        with tqdm(total=args.train_samples_qty) as pbar:
            loader = SessionDataLoader(train_dataset, batch_size=batch_size)
            print('\nget loader success\n')
            for feat, target, mask in loader:
                   
                real_mask = np.ones((batch_size, 1))
                for elt in mask:
                    real_mask[elt, :] = 0
                
                hidden_states = get_states(model_to_train)[0]
                hidden_states = np.multiply(real_mask, hidden_states)
                hidden_states = np.array(hidden_states, dtype=np.float32)
                model_to_train.layers[2].reset_states(hidden_states)   # layers[gru那層]
                #print('\nhidden states\n')
                input_oh = to_categorical(feat, num_classes=args.train_n_items) # loader.n_items
                input_oh = np.expand_dims(input_oh, axis=1)

                target_oh = to_categorical(target, num_classes=args.train_n_items) # 29474 37098

                tr_loss = model_to_train.train_on_batch(input_oh, target_oh)

                pbar.set_description("Epoch {0}. Loss: {1:.5f}".format(epoch, tr_loss))
                pbar.update(loader.done_sessions_counter)
            
        if save_weights:
            print("Saving weights...")
            model_to_train.save('./GRU4REC_126_{}.h5'.format(epoch))
        
        (rec, rec_k), (mrr, mrr_k) = get_metrics(model_to_train, args, train_dataset.itemmap)

        print("\t - Recall@{} epoch {}: {:5f}".format(rec_k, epoch, rec))
        print("\t - MRR@{}    epoch {}: {:5f}".format(mrr_k, epoch, mrr))
        print("\n")
        
        
        
            
if __name__ == '__main__':
    print('exicute..')
    parser = argparse.ArgumentParser(description='Keras GRU4REC: session-based recommendations')
    parser.add_argument('--resume', type=str, help='stored model path to continue training')
    parser.add_argument('--train-path', type=str, default='rsc15_4_train_tr.txt') #../../processData/
    parser.add_argument('--dev-path', type=str, default='rsc15_4_train_valid.txt')
    parser.add_argument('--test-path', type=str, 
default='rsc15_4_test.txt')
    parser.add_argument('--batch-size', type=str, default=1000)#512 364 438
    args = parser.parse_args()

    args.train_data = pd.read_csv(args.train_path, sep='\t', dtype={'ItemId': np.int64})#[0:400000]
    args.dev_data   = pd.read_csv(args.dev_path, sep='\t', dtype={'ItemId': np.int64})
    args.test_data  = pd.read_csv(args.test_path, sep='\t', dtype={'ItemId': np.int64})
    
    args.train_n_items = 8941 #9026(12540)9088(125)8829(127)13350(12240) 9029(12360) 6882(1_f20p,3_f60p # 8979 # 15246 # 37098   # len(args.train_data['ItemId'].unique()) - 1 # 29474 #

    args.train_samples_qty = len(args.train_data['SessionId'].unique()) + 1
    args.test_samples_qty = len(args.test_data['SessionId'].unique()) + 1
#     print('a lo ha')
    if args.resume:
        try:
            model = keras.models.load_model(args.resume)
            print("Model checkpoint '{}' loaded!".format(args.resume))
        except OSError:
            print("Model checkpoint could not be loaded. Training from scratch...")
            model = create_model(args)
#             model = create_vae_model(args)
    else:
        print(args.train_path)
        model = create_model(args)
#         model = create_vae_model(args)
    
#     transfer_learning(model, args, save_weights=True)        
    train_model(model, args, save_weights=True)

