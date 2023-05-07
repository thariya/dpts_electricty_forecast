import typing
from typing import Tuple
import collections
import json
import os
import logging

import torch
from torch import nn
from torch import optim
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as tf
from torch.autograd import Variable

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

print(torch.__version__)

train_file="Ausgrid_Newcastle_2011-07-01_2012-06-30.csv"
test_file="Ausgrid_Newcastle_2012-07-01_2013-06-30.csv"
device=torch.device("cpu")

def setup_log(tag='VOC_TOPICS'):
    # create logger
    logger = logging.getLogger(tag)
    # logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def save_or_show_plot(file_nm: str, save: bool):
    if save:
        plt.savefig(os.path.join(os.path.dirname(__file__), "plots", file_nm))
    else:
        plt.show()

def numpy_to_tvar(x):
    return Variable(torch.from_numpy(x).type(torch.FloatTensor).to(device))

class TrainConfig(typing.NamedTuple):
    T: int
    train_size: int
    batch_size: int
    loss_func: typing.Callable
class TrainData(typing.NamedTuple):
    feats: np.ndarray
    targs: np.ndarray
DaRnnNet = collections.namedtuple("DaRnnNet", ["encoder", "decoder", "enc_opt", "dec_opt"])


def init_hidden(x, hidden_size: int):
    """
    Train the initial value of the hidden state:
    https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
    """
    return Variable(torch.zeros(1, x.size(0), hidden_size))


class Encoder(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, T: int):
        """
        input size: number of underlying factors (81)
        T: number of time steps (10)
        hidden_size: dimension of the hidden state
        """
        
        super(Encoder, self).__init__()
        
        time_kernel_sizes=[3,5,7,11,13]       
        cnns = []
        for k in time_kernel_sizes:
            seq = nn.Sequential(nn.Conv1d(input_size, input_size, k, padding=int((k-1)/2)), nn.Tanh())
            cnns.append(seq)
        self.cnns = nn.ModuleList(cnns)
        
        
        self.input_size = input_size*len(time_kernel_sizes)
        self.inp_size=input_size
        self.hidden_size = hidden_size
        self.T = T

        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.attn_conv = nn.Linear(in_features=2 * hidden_size + T - 1, out_features=1)
        self.attn_cos = nn.Linear(in_features=2 * hidden_size + T - 1, out_features=1)
        self.regr_linear = nn.Linear(in_features=self.input_size+self.inp_size, out_features=input_size)
        
        time_kernel_sizes=[3,5,7,11,13]       
        cnns = []
        for k in time_kernel_sizes:
            seq = nn.Sequential(nn.Conv1d(input_size, input_size, k, padding=int((k-1)/2)), nn.Tanh())
            cnns.append(seq)
        self.cnns = nn.ModuleList(cnns) 


    def forward(self, input_data):
        # input_data: (batch_size, T - 1, input_size)
        input_weighted = Variable(torch.zeros(input_data.size(0), self.T - 1, int(self.inp_size))).to(device)
        input_encoded = Variable(torch.zeros(input_data.size(0), self.T - 1, self.hidden_size)).to(device)
        # hidden, cell: initial states with dimension hidden_size
        hidden = init_hidden(input_data, self.hidden_size).to(device)  # 1 * batch_size * hidden_size
        cell = init_hidden(input_data, self.hidden_size).to(device)  

        conv_input=input_data.permute(0, 2, 1)  # (batch_size,input_size, T - 1)
        tmp_input = [cnn(conv_input) for cnn in self.cnns]
        modified_data = torch.cat(tuple(tmp_input), dim=1)        

        for t in range(self.T - 1):          
             
            ##Convolution section
            # concatenate the hidden states with each predictor
            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           modified_data), dim=2)  # batch_size * input_size * (2*hidden_size + T - 1)
            # Get attention weights            
            x = self.attn_conv(x.view(-1, self.hidden_size * 2 + self.T - 1))  # (batch_size * input_size) * 1
            
            ##Cosine section
            #concatenate the hidden states with each predictor
            y = torch.cat((hidden.repeat(self.inp_size, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.inp_size, 1, 1).permute(1, 0, 2),
                           conv_input), dim=2)  # batch_size * input_size * (2*hidden_size + T - 1)
            #Get attention weights            
            y = torch.cos(self.attn_cos(y.view(-1, self.hidden_size * 2 + self.T - 1)))  # (batch_size * input_size) * 1

            xy=torch.cat((x.view(-1, self.input_size),y.view(-1, self.inp_size)),dim=1)

            #Reducing the number of inputs back
            attn_weights = self.regr_linear(xy)  # (batch_size, input_size) 
            
            # Softmax the attention weights
            attn_weights = tf.softmax(attn_weights, dim=1)  # (batch_size, input_size)
            
            
            
            # LSTM
            weighted_input = torch.mul(attn_weights, input_data[:, t, :])  # (batch_size, input_size)
            
            #weighted_input=torch.cos(attn_linear(input_data[:, t, :]))
            # Fix the warning about non-contiguous memory
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]
            # Save output
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden

        return input_weighted, input_encoded

        
class Decoder(nn.Module):

    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int, T: int, out_feats=1):
        super(Decoder, self).__init__()

        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size,
                                                  encoder_hidden_size),
                                        nn.Tanh(),
                                        nn.Linear(encoder_hidden_size, 1))
        self.lstm_layer = nn.LSTM(input_size=out_feats, hidden_size=decoder_hidden_size)
        self.fc = nn.Linear(encoder_hidden_size + out_feats, out_feats)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, out_feats)

        self.fc.weight.data.normal_()

    def forward(self, input_encoded, y_history):
        # input_encoded: (batch_size, T - 1, encoder_hidden_size)
        # y_history: (batch_size, (T-1))
        # Initialize hidden and cell, (1, batch_size, decoder_hidden_size)
        hidden = init_hidden(input_encoded, self.decoder_hidden_size).to(device)
        cell = init_hidden(input_encoded, self.decoder_hidden_size).to(device)
        context = Variable(torch.zeros(input_encoded.size(0), self.encoder_hidden_size))

        
        for t in range(self.T - 1):
            # (batch_size, T, (2 * decoder_hidden_size + encoder_hidden_size))
            x = torch.cat((hidden.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           input_encoded), dim=2)
            # softmax on the computed attention weights
            x = tf.softmax(
                    self.attn_layer(
                        x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)
                    ).view(-1, self.T - 1),
                    dim=1)  # (batch_size, T - 1)

            # compute context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]  # (batch_size, encoder_hidden_size)

            y_tilde = self.fc(torch.cat((context, y_history[:, t]), dim=1))  # (batch_size, out_size)
            # LSTM
            self.lstm_layer.flatten_parameters()
            _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
            hidden = lstm_output[0]  # 1 * batch_size * decoder_hidden_size
            cell = lstm_output[1]  # 1 * batch_size * decoder_hidden_size

        # final output
        return self.fc_final(torch.cat((hidden[0], context), dim=1))



logger = setup_log()
logger.info(f"Using computation device: {device}")


def preprocess_data(dat, col_names) -> Tuple[TrainData, StandardScaler]:
    scale = StandardScaler().fit(dat)
    proc_dat = scale.transform(dat)
    print(col_names,"col_names")
    mask = np.ones(proc_dat.shape[1], dtype=bool)
    dat_cols = list(dat.columns)
    for col_name in col_names:
        mask[dat_cols.index(col_name)] = False

    feats = proc_dat[:, mask]
    targs = proc_dat[:, ~mask]
    # print(targs)

    return TrainData(feats, targs), scale


def da_rnn(train_length,train_data: TrainData, n_targs: int, encoder_hidden_size=64, decoder_hidden_size=64,
           T=48, learning_rate=0.01, batch_size=128):

    train_cfg = TrainConfig(T, int(train_length), batch_size, nn.MSELoss())
    logger.info(f"Training size: {train_cfg.train_size:d}.")

    enc_kwargs = {"input_size": train_data.feats.shape[1], "hidden_size": encoder_hidden_size, "T": T}
    encoder = Encoder(**enc_kwargs).to(device)


    dec_kwargs = {"encoder_hidden_size": encoder_hidden_size,
                  "decoder_hidden_size": decoder_hidden_size, "T": T, "out_feats": n_targs}
    decoder = Decoder(**dec_kwargs).to(device)


    encoder_optimizer = optim.Adam(
        params=[p for p in encoder.parameters() if p.requires_grad],
        lr=learning_rate)
    decoder_optimizer = optim.Adam(
        params=[p for p in decoder.parameters() if p.requires_grad],
        lr=learning_rate)
    da_rnn_net = DaRnnNet(encoder, decoder, encoder_optimizer, decoder_optimizer)

    return train_cfg, da_rnn_net


def train(net: DaRnnNet, train_data: TrainData, t_cfg: TrainConfig, n_epochs=10, save_plots=False):
    iter_per_epoch = int(np.ceil(t_cfg.train_size * 1. / t_cfg.batch_size))
    iter_losses = np.zeros(n_epochs * iter_per_epoch)
    epoch_losses = np.zeros(n_epochs)
    logger.info(f"Iterations per epoch: {t_cfg.train_size * 1. / t_cfg.batch_size:3.3f} ~ {iter_per_epoch:d}.")

    n_iter = 0

    for e_i in range(n_epochs):
        perm_idx = np.random.permutation(t_cfg.train_size - t_cfg.T)

        for t_i in range(0, t_cfg.train_size, t_cfg.batch_size):
            batch_idx = perm_idx[t_i:(t_i + t_cfg.batch_size)]

            feats, y_history, y_target = prep_train_data(batch_idx, t_cfg, train_data)

            loss = train_iteration(net, t_cfg.loss_func, feats, y_history, y_target)
            iter_losses[e_i * iter_per_epoch + t_i // t_cfg.batch_size] = loss
            n_iter += 1

            adjust_learning_rate(net, n_iter)

        epoch_losses[e_i] = np.mean(iter_losses[range(e_i * iter_per_epoch, (e_i + 1) * iter_per_epoch)])

        if e_i % 10 == 0:
            y_test_pred = predict(net, train_data,
                                  t_cfg.train_size, t_cfg.batch_size, t_cfg.T,
                                  on_train=False)
            # TODO: make this MSE and make it work for multiple inputs
            val_loss = y_test_pred - train_data.targs[t_cfg.train_size:]
            logger.info(f"Epoch {e_i:d}, train loss: {epoch_losses[e_i]:3.3f}, val loss: {np.mean(np.abs(val_loss))}.")
            y_train_pred = predict(net, train_data,
                                   t_cfg.train_size, t_cfg.batch_size, t_cfg.T,
                                   on_train=True)
            plt.figure()
            plt.plot(range(1, 1 + len(train_data.targs)), train_data.targs,
                     label="True")
            plt.plot(range(t_cfg.T, len(y_train_pred) + t_cfg.T), y_train_pred,
                     label='Predicted - Train')
            plt.plot(range(t_cfg.T + len(y_train_pred), len(train_data.targs) + 1), y_test_pred,
                     label='Predicted - Test')
            plt.legend(loc='upper left')

    return iter_losses, epoch_losses


def prep_train_data(batch_idx: np.ndarray, t_cfg: TrainConfig, train_data: TrainData):
    feats = np.zeros((len(batch_idx), t_cfg.T - 1, train_data.feats.shape[1]))
    y_history = np.zeros((len(batch_idx), t_cfg.T - 1, train_data.targs.shape[1]))
    y_target = train_data.targs[batch_idx + t_cfg.T]

    for b_i, b_idx in enumerate(batch_idx):
        b_slc = slice(b_idx, b_idx + t_cfg.T - 1)
        feats[b_i, :, :] = train_data.feats[b_slc, :]
        y_history[b_i, :] = train_data.targs[b_slc]

    return feats, y_history, y_target


def adjust_learning_rate(net: DaRnnNet, n_iter: int):
    if n_iter % 10000 == 0 and n_iter > 0:
        for enc_params, dec_params in zip(net.enc_opt.param_groups, net.dec_opt.param_groups):
            enc_params['lr'] = enc_params['lr'] * 0.9
            dec_params['lr'] = dec_params['lr'] * 0.9


def train_iteration(t_net: DaRnnNet, loss_func: typing.Callable, X, y_history, y_target):

    t_net.enc_opt.zero_grad()
    t_net.dec_opt.zero_grad()

    input_weighted, input_encoded = t_net.encoder(numpy_to_tvar(X))
    y_pred = t_net.decoder(input_encoded, numpy_to_tvar(y_history))

    y_true = numpy_to_tvar(y_target)

    if (len(y_pred)!=0) & (len(y_true)!=0):
     loss = loss_func(y_pred, y_true)
     loss.backward()

     t_net.enc_opt.step()
     t_net.dec_opt.step()

     return loss.item()
    else:
        return 0


def predict(t_net: DaRnnNet, t_dat: TrainData, train_size: int, batch_size: int, T: int, on_train=False):
    out_size = t_dat.targs.shape[1]
    if on_train:
        y_pred = np.zeros((train_size - T + 1, out_size))
    else:
        y_pred = np.zeros((t_dat.feats.shape[0] - train_size, out_size))

    for y_i in range(0, len(y_pred), batch_size):
        y_slc = slice(y_i, y_i + batch_size)
        batch_idx = range(len(y_pred))[y_slc]
        b_len = len(batch_idx)
        X = np.zeros((b_len, T - 1, t_dat.feats.shape[1]))
        y_history = np.zeros((b_len, T - 1, t_dat.targs.shape[1]))

        for b_i, b_idx in enumerate(batch_idx):
            if on_train:
                idx = range(b_idx, b_idx + T - 1)
            else:
                idx = range(b_idx + train_size - T, b_idx + train_size - 1)

            X[b_i, :, :] = t_dat.feats[idx, :]
            y_history[b_i, :] = t_dat.targs[idx]

        y_history = numpy_to_tvar(y_history)
        _, input_encoded = t_net.encoder(numpy_to_tvar(X))
        y_pred[y_slc] = t_net.decoder(input_encoded, y_history).cpu().data.numpy()

    return y_pred


save_plots = True
debug = False

raw_data_1 = pd.read_csv(train_file, usecols=["Total_consumption","Index","Air_temperature","Relative_humidity"], nrows=100 if debug else None)
raw_data_2 = pd.read_csv(test_file, usecols=["Total_consumption","Index","Air_temperature","Relative_humidity"], nrows=100 if debug else None)

frames=[raw_data_1,raw_data_2]
raw_data=pd.concat(frames)
logger.info(f"Shape of data: {raw_data.shape}.\nMissing in data: {raw_data.isnull().sum().sum()}.")
targ_cols = ("Total_consumption",)
data, scaler = preprocess_data(raw_data, targ_cols)

da_rnn_kwargs = {"batch_size": 128, "T":49}
config, model = da_rnn(len(raw_data_1),data, n_targs=len(targ_cols), learning_rate=.001, **da_rnn_kwargs)
iter_loss, epoch_loss = train(model, data, config, n_epochs=50, save_plots=save_plots)
final_y_pred = predict(model, data, config.train_size, config.batch_size, config.T)

plt.figure()
plt.semilogy(range(len(iter_loss)), iter_loss)


plt.figure()
plt.semilogy(range(len(epoch_loss)), epoch_loss)


plt.figure()
plt.plot(final_y_pred, label='Predicted')
plt.plot(data.targs[config.train_size:], label="True")
plt.legend(loc='upper left')


df_new=pd.DataFrame()
true_list=list((data.targs[config.train_size:]))
list_true=[]
for i in range(0,len(true_list)):
    list_true.append(list(true_list[i])[0])


list_true=[x * (np.std(raw_data_1["Total_consumption"]))+np.mean(raw_data_1["Total_consumption"]) for x in list_true]
final_y_pred=[x * (np.std(raw_data_1["Total_consumption"]))+np.mean(raw_data_1["Total_consumption"]) for x in final_y_pred]
final_y_pred=[x[0] for x in final_y_pred]

df_new["True"]=list_true
df_new["Predicted"]=list(final_y_pred)
zip_object = zip(list_true,list(final_y_pred))
MAE_overall=[]
for list1_i, list2_i in zip_object:
    MAE_overall.append(np.abs(list1_i-list2_i)) # append each difference to list

df_new['MAE']=MAE_overall

con = df_new["True"]

peak_weight = 1
weights_all = []
for count in range(0, int(np.round(len(con) / 48))):
    weights_tr = np.zeros(48)

    mul = count * 48
    y_daily = con[mul:mul + 48]

    peak_threshold_train = np.max(y_daily)

    index_tr = np.where(y_daily == peak_threshold_train)[0]

    weights_tr[index_tr] = peak_weight
    weights_all.extend(weights_tr)

print(len(weights_all), len(con))
if len(weights_all) == len(con):
    df_new['Peak_daily_max_48'] = weights_all

else:
    n = len(weights_all) - len(con)    
    weights_all = weights_all[:-n]

    df_new['Peak_daily_max_48'] = weights_all
zip_object = zip(list(df_new['True']), list(df_new['Predicted']))
MAE_overall = []
for list1_i, list2_i in zip_object:
    MAE_overall.append(np.abs(list1_i - list2_i))  # append each difference to list

df_new['MAE'] = MAE_overall
df_peak = df_new.loc[df_new['Peak_daily_max_48'] == 1]
print(np.mean(df_peak['MAE']), "peak mae")
print(np.mean(df_new['MAE']), "overall mae")
df_new.to_csv("results_scaled_gru_12.csv")

torch.save(model.encoder.state_dict(), "encoder.torch")
torch.save(model.decoder.state_dict(), "decoder.torch")
