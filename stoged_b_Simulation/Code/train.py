import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from datetime import datetime
import logging
import time
import scipy.io as io
import seaborn as sns
import matplotlib as mplb
import matplotlib.pyplot as plt
import random
import scipy
import pickle
from collections import namedtuple


import sys, pathlib, os
# root = pathlib.Path(__file__).parent.parent.parent
# sys.path.insert(0,str(root))
from stoged_b_Simulation.utils.models import CNN
from stoged_b_Simulation.utils.generate_signals import generate_periodic_signals_all, generate_CS2_signals_all, generate_periodic_signals_all_V2
from stoged_b_Simulation.utils.logger import setlogger
from stoged_b_Simulation.utils.DatasetBase import myDataset
from stoged_b_Simulation.utils.mysummary import summary


# --------------------------------Initialize----------------------------------------------
data_load_map = {0: generate_periodic_signals_all, 1: generate_CS2_signals_all, 2: generate_periodic_signals_all_V2}
data_name_dict = {0: 'periodic', 1: 'CS2', 2: 'periodicV2'}
# ----------------------------------Preparation--------------------------------------------


class main(object):
    def __init__(self):
        self.GetParams()
        self.Preparation()
        self.setup()
        self.Train()
        self.Plot()

    def GetParams(self):
        # hyperparameters-dataset
        Params = {}
        Params['data_select'] = 2  # 0: periodic signals, 1: CS2 signals, 2: periodicV2 signals
        Params['L'] = 1024
        Params['N_sample'] = int(1e4)
        Params['test_size'] = 0.3
        # hyperparameters-model

        # hyperparameters-train
        Params['batch_size'] = 64
        Params['max_epoch'] = 100
        Params['learning_rate'] = 0.001
        Params['weight_decay'] = 0.001
        Params['gamma'] = 0.99
        Params['print_step'] = 20
        # hyperparameters-save
        Params['checkpoint_dir'] = '../checkpoint'
        Params['save_model'] = True
        Params['save_N_data'] = 100

        # post process
        Params['data_name'] = data_name_dict[Params['data_select']]
        temp = namedtuple("Params", Params.keys())
        self.Params = temp(**Params) # self.Params._asdict()

    def Preparation(self):
        # Initialize the directory to save the model
        sub_dir = self.Params.data_name + '-' + str(self.Params.L) + '-' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
        self.save_dir = os.path.join(self.Params.checkpoint_dir, sub_dir).replace('\\', '/')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # set the logger
        setlogger(os.path.join(self.save_dir, 'training.log'))

        # save the args
        for k, v in self.Params._asdict().items():
            logging.info("<args> {}: {}".format(k, v))

    def setup(self):
        # load data
        generate_signals = data_load_map[self.Params.data_select]

        signals, labels, label_name = generate_signals(L=self.Params.L, N_sample=int(self.Params.N_sample))


        signals = np.expand_dims(signals, axis=1)
        # build dataloader
        index = \
            list(StratifiedShuffleSplit(n_splits=1, test_size=self.Params.test_size, random_state=0)
                 .split(np.ones(labels.shape[0]), labels))[0]
        # print(pd.array(labels[index[0]]).value_counts())
        self.label_name = label_name
        self.dataset, self.dataloader = {}, {}
        for phase in ['train', 'val']:
            index_phase = index[0] if phase == 'train' else index[1]
            self.dataset[phase] = myDataset([signals[index_phase], labels[index_phase]])
            self.dataloader[phase] = torch.utils.data.DataLoader(
                self.dataset[phase], batch_size=self.Params.batch_size, shuffle=True)

        logging.info('train_datasize: {} | val_datasize: {}'.format(len(self.dataset['train']),
                                                                    len(self.dataset['val'])))
        # build CNN model
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # self.device = torch.device("cpu")
        self.model = CNN(in_channel=1, out_channel=len(label_name)).to(self.device)
        info = summary(self.model, self.dataset['train'][0][0].shape, batch_size=-1, device=str(self.device))
        for item in info.split('\n'):
            logging.info(item)

        # build optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.Params.learning_rate, weight_decay=self.Params.weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.Params.gamma)

        # Load the checkpoint
        self.start_epoch = 0

        # confusion matrix
        self.c_matrix = {phase: np.zeros([self.Params.max_epoch,len(set(labels)),len(set(labels))]) for phase in ['train', 'val']}

        # save part data
        collected_data_for_analysis = {'train':{}, 'val':{}, 'label_name':self.label_name}
        for phase in ['train', 'val']:
            labels = pd.array(self.dataset[phase].labels.numpy())
            for item in labels.value_counts().keys():
                index = np.where(labels == item)[0]
                collected_data_for_analysis[phase][str(item)] = [temp.squeeze().numpy() for temp in self.dataset[phase][index[:self.Params.save_N_data]]]
        pickle.dump(collected_data_for_analysis, open(os.path.join(self.save_dir, 'collected_data_for_analysis.pkl'), 'wb'))
        # with open(os.path.join(self.save_dir, 'collected_data_for_analysis.pkl'), 'rb') as f:
        #     a = pickle.load(f)


    def Train(self):
        Params = self.Params
        best_acc = 0.0
        time_start_train = time.time()
        step_start = time.time()
        self.Records = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], 'best_epoch': 0}
        self.MinorRecords = {"train_loss": [], "train_acc": [],"val_loss": [], "val_acc": []}
        for epoch in range(self.start_epoch, Params.max_epoch):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, Params.max_epoch - 1) + '-' * 5)
            # Update the learning rate
            if self.lr_scheduler is not None:
                # self.lr_scheduler.step(epoch)
                logging.info(
                    'current lr: {}'.format([float('%.8f' % item) for item in self.lr_scheduler.get_last_lr()]))
            else:
                logging.info('current lr: {}'.format([self.Params.learning_rate, ]))

            for phase in ['train', 'val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc, epoch_loss = 0, 0
                batch_acc, batch_loss,batch_count = 0, 0, 0
                # Set models to train mode or test mode
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                for batch_idx, (inputs, labels_) in enumerate(self.dataloader[phase]):
                    inputs = inputs.to(self.device)
                    if type(labels_) == list:
                        labels = labels_[0].type(torch.int64).to(self.device)
                    else:
                        labels = labels_.type(torch.int64).to(self.device)
                    # Do the learning process, in val, we do not care about the gradient for relaxing
                    with torch.set_grad_enabled(phase == 'train'):
                        logits = self.model(inputs)
                        loss = self.criterion(logits, labels)
                        if phase == 'train':
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                    # Calculate the accuracy
                    pred = logits.argmax(dim=1)
                    correct = torch.eq(pred, labels).float().sum().item()
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_acc += correct

                    # confusion matrix
                    for i, j in zip(labels.detach().cpu().numpy(), pred.detach().cpu().numpy()):
                        self.c_matrix[phase][epoch][i][j] += 1


                    # Record minor information
                    self.MinorRecords['%s_loss'% phase].append(loss.item())
                    self.MinorRecords['%s_acc'% phase].append(correct / inputs.size(0))

                    # Calculate the information
                    batch_acc += correct
                    batch_loss += loss.item() * inputs.size(0)
                    batch_count += inputs.size(0)
                    # Print the training information
                    if phase == 'train' and (batch_idx + 1) % Params.print_step == 0:
                        batch_loss = batch_loss / batch_count
                        batch_acc = batch_acc / batch_count
                        temp_time = time.time()
                        train_time = temp_time - step_start
                        step_start = temp_time
                        batch_time = train_time / Params.print_step
                        sample_per_sec = 1.0 * batch_count / train_time
                        temp_info = f'<tempinfo> Epoch: {epoch:3d} [{(batch_idx + 1)*len(inputs):4d}/{len(self.dataloader[phase].dataset):4d}],     step: {batch_idx + 1:3d}'\
                                    + f'     {sample_per_sec:.1f} samples/sec     {batch_time:.4f} sec/batch     Train Acc: {batch_acc * 100:.2f},  Train Loss: { batch_loss:.4f}'
                        logging.info(temp_info)
                        batch_acc, batch_loss,batch_count = 0, 0, 0

                # Print the train and val information via each epoch
                epoch_loss /= len(self.dataloader[phase].dataset)
                epoch_acc /= len(self.dataloader[phase].dataset)
                temp_info = f'<info> Epoch: {epoch:d},  Cost {time.time() - epoch_start:.4f} sec,' + \
                            f'  {phase:s}-Acc: {epoch_acc * 100:.4f},  {phase:s}-Loss: {epoch_loss:.4f}'
                logging.info(temp_info)

                # Record major information
                self.Records["%s_loss" % phase].append(epoch_loss)
                self.Records["%s_acc" % phase].append(epoch_acc)

                # save the models
                if phase == 'val':
                    if epoch_acc >= best_acc:
                        best_acc = epoch_acc
                        self.Records['best_epoch'] = epoch + 1
                        logging.info("save best epoch {}, best acc {:.4f}".format(epoch+1, epoch_acc))
                        save_best_data_dir = os.path.join(self.save_dir,
                                                          'epoch{}-acc{:.4f}-best_model.pth'.format(epoch+1,
                                                                                                    epoch_acc * 100))
                        save_best_data = {'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
                                          'optimizer': self.optimizer.state_dict()}

                    if epoch == Params.max_epoch - 1:
                        if Params.save_model:
                            # save the best models according to the val accuracy
                            torch.save(save_best_data, save_best_data_dir)
                            # save the final models
                            logging.info("save final epoch {}, final acc {:.4f}".format(epoch+1, epoch_acc))
                            save_data = {'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
                                          'optimizer': self.optimizer.state_dict()}
                            torch.save(save_data,
                                       os.path.join(self.save_dir,
                                                    'epoch{}-acc{:.4f}-final_model.pth'.format(epoch+1, epoch_acc * 100)))
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # After train

        # stacks Record list to numpy when finished train process
        for k in self.Records.keys():
            self.Records[k] = np.array(self.Records[k])
        for k in self.MinorRecords.keys():
            self.MinorRecords[k] = np.array(self.MinorRecords[k])
        io.savemat(os.path.join(self.save_dir, "Records.mat"), self.Records)
        io.savemat(os.path.join(self.save_dir, "MinorRecords.mat"), self.MinorRecords)
        io.savemat(os.path.join(self.save_dir, "confusion_matrix.mat"), self.c_matrix)

        # log the best and final acc and loss
        info = "max train acc in epoch {:2d}: {:10.6f}\n".format(self.Records['train_acc'].argmax() + 1,
                                                                 self.Records['train_acc'].max()) \
               + "max val acc in epoch {:2d}: {:10.6f}\n".format(self.Records['val_acc'].argmax() + 1,
                                                                 self.Records['val_acc'].max()) \
               + "final train acc: %.6f\n final val acc: %.6f\n" \
               % (self.Records['train_acc'][-1], self.Records['val_acc'][-1])
        for item in info.split('\n'):
            logging.info(item)
        with open(os.path.join(self.save_dir, 'acc output.txt'), 'w') as f:
            f.write(info)

        # training time
        logging.info("<training time>: {:.3f}".format(time.time() - time_start_train))


    def Plot(self):
        #set color
        current_cmap = sns.color_palette("husl", 10)
        sns.set(style="white")
        sns.set(style="ticks", context="notebook", font='Times New Roman', palette=current_cmap, font_scale=1.5)
        mplb.rcParams['font.size'] = 12
        mplb.rcParams['font.family'] = 'Times New Roman'
        mplb.rcParams['figure.dpi'] = 300

        # make dir
        self.save_dir_sub = os.path.join(self.save_dir, "postprosess")
        if not os.path.exists(self.save_dir_sub):
            os.makedirs(self.save_dir_sub)

        #plot confusion matrix
        for phase in ['train','val']:
            f, ax = plt.subplots(figsize=(10,8))
            sns.heatmap(self.c_matrix[phase][-1].astype(int), annot=True, ax=ax, fmt="d",cmap="RdBu_r",linewidths=0.3,annot_kws={'size':12})
            ax.invert_yaxis()
            ax.set_xticklabels(self.label_name)
            ax.set_yticklabels(self.label_name)
            ax.set_xlabel('predict')
            ax.set_ylabel('true')
            ax.set_title('confusion matrix: %s'%phase)
            f.savefig(os.path.join(self.save_dir_sub, f"confusion_matrix_{phase:s}-epoch{len(self.c_matrix[phase]):d}.jpg"))

        # plot Records
        Ls = ['-', '--', '-.', ':']
        Markers = ['.',',','o','v','^','<','>','1','2','3','4','8','s',]
        epochs = np.arange(self.Records['train_loss'].shape[0])+1
        fig = plt.figure(figsize=[10,8],dpi=600)
        ax1 = fig.add_subplot(2,1,1) # loss
        lines = ax1.plot(epochs,self.Records['train_loss'],"b",markersize=2,linewidth=2)
        # for i in range(len(lines)-1):
        #     lines[i].set(ls = Ls[i%len(Ls)],alpha = 0.6*(1-i/(len(lines)-1)), marker = Markers[i%len(Markers)])
        lines = ax1.plot(epochs, self.Records['val_loss'], "r", markersize=2,linewidth=2)
        # for i in range(len(lines)-1):
        #     lines[i].set(ls = Ls[i%len(Ls)],alpha = 0.6*(1-i/(len(lines)-1)), marker = Markers[i%len(Markers)])
        # ax1.legend(['train loss %d'%i for i in range(self.Records['train_loss'].shape[1])] + ['val loss %d'%i for i in range(self.Records['val_loss'].shape[1])],loc =5)
        ax1.legend(['train loss', 'val loss'])
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        # ax1.grid()
        ax2 = fig.add_subplot(2, 1, 2) # acc
        ax2.plot(epochs, self.Records['train_acc']*100,"b-.d", epochs, self.Records['val_acc']*100,"r-.d",markersize=2,linewidth=2)
        ax2.legend(['train acc', 'val acc'],loc =5)
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('acc')
        # ax2.set_ylim([80,100])
        fig.tight_layout()
        # ax2.grid()
        fig.savefig(os.path.join(self.save_dir_sub,"loss_acc.jpg"))

        # plot MinorRecords
        steps_t = np.arange(self.MinorRecords['train_loss'].shape[0]) + 1
        steps_t = steps_t / len(steps_t) * self.Params.max_epoch
        steps_v = np.arange(self.MinorRecords['val_loss'].shape[0]) + 1
        steps_v = steps_v / len(steps_v) * self.Params.max_epoch
        fig = plt.figure(figsize=[10, 8], )
        ax1 = fig.add_subplot(2, 1, 1) # acc
        lines = ax1.plot(steps_t, self.MinorRecords['train_loss'], "b",
                         steps_v, self.MinorRecords['val_loss'], "r",
                         markersize=3, linewidth=2, markevery= int(0.05 * len(steps_t)))
        # for i in range(len(lines) - 1):
        #     lines[i + 1].set_alpha(0.6 * (1 - i / (len(lines) - 1)))
        # ax1.grid()
        ax1.legend(['train loss', 'val loss'])
        ax1.set_xlabel('epoch', fontfamily='monospace')
        ax1.set_ylabel('loss', fontfamily='monospace')
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(steps_t, self.MinorRecords['train_acc'] * 100, "b",
                 steps_v, self.MinorRecords['val_acc'] * 100, "r",
                 markersize=3, linewidth=2, markevery= int(0.05 * len(steps_t)))
        ax2.legend(['train acc', 'val acc'], loc=5)
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('acc')
        # ax2.set_ylim([80,100])
        fig.tight_layout()
        # ax2.grid()
        fig.savefig(os.path.join(self.save_dir_sub, "Minor_loss_acc.jpg"))

        # plt data spectrum
        self.collected_data_for_spectrum = {}
        index = np.arange(len(self.dataset['train']))
        random.shuffle(index)
        for i in index[:100]:
            item,label = self.dataset['train'][i]
            if label.item() not in self.collected_data_for_spectrum.keys():
                self.collected_data_for_spectrum[str(label.item())] = item.numpy().squeeze()

        A = np.array(list(self.collected_data_for_spectrum.values()))
        Z = []
        for item in self.collected_data_for_spectrum.values():
            z= scipy.fft.fft(item-item.mean())
            Z.append(abs(z))
        Z = np.array(Z)*z.shape[-1]
        f = np.arange(Z.shape[-1])/Z.shape[-1]*2
        fig = plt.figure(figsize=[8, 5])
        ax = fig.add_subplot()
        ax.plot(f, Z.T)
        ax.set_xlim(0, 1)
        ax.legend([self.label_name[int(item)] for item in list(self.collected_data_for_spectrum.keys())])
        fig.savefig(os.path.join(self.save_dir_sub, "data_spectrum.jpg"))
        fig = plt.figure(figsize=[8, 5])
        ax = fig.add_subplot()
        ax.plot(range(A.squeeze().shape[1]), A.squeeze().T)
        ax.set_xlim(0, 300)
        ax.legend([self.label_name[int(item)] for item in list(self.collected_data_for_spectrum.keys())])
        fig.savefig(os.path.join(self.save_dir_sub, "time.jpg"))




if __name__ == '__main__':
    main()
