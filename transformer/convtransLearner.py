import torch
import sys
import numpy as np
import time
from model_convtrans import  *
from utils import *
# teee
class convtransLearner(object):
    def __init__(self, **kwargs):
        src_vocab = kwargs['src_vocab']
        learning_rate = kwargs['lr']
        weight_decay = kwargs['wd']
        dropout = kwargs['dropout']
        batch_size = kwargs['bs']
        n_class = kwargs['n_class']
        N, d_model, d_ff, h= kwargs['N'], kwargs['d_model'], kwargs['d_ff'], kwargs['h']
        print ('learning rate is: ', learning_rate)

        self.filename = kwargs['record_file']
        self.model_name = kwargs['model_name']
        if kwargs['test']:
            self.model = torch.load(self.model_name)
        else:
            self.model = make_model(src_vocab, n_class, N=N, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout)
            print ('weight_decay={}'.format(weight_decay))

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            self.batch_size = batch_size

    def train(self, num_epoch, train_loader, valid_loader):
        verbose=False
        self.file_obj = open(self.filename, 'w')
        self.file_obj.write('Training:\n')
        self.file_obj.close()
        best_score = -10000000000
        for epoch in range(num_epoch):
            print('epoch=', epoch)
            self.model.train()
            start = time.time()
            total_loss = 0
            for batch_train_x, batch_train_y in train_loader:
                batch_train_x = batch_train_x.cuda()
                batch_train_y = batch_train_y.cuda()
                batch_start_time = time.time()
                self.optimizer.zero_grad()
                loss_train = self.model.loss(batch_train_x, batch_train_y)
                loss_train.backward()
                total_loss += float(loss_train.data)
                self.optimizer.step()
                batch_cost_time = time.time() - batch_start_time
                print('fake print=', batch_cost_time)
                if verbose:
                    sys.stdout.write(str(batch_cost_time))
                    sys.stdout.flush()

            auc = self.eval(valid_loader)
            seconds = time.time() - start
            output = ('Epoch={}, time {:.4f}, train loss {:.4f} '.format(epoch, seconds, total_loss)
                    +  'valid auc {:.4f}'.format(auc))
            print (output)
            with open(self.filename, 'a') as outfile:
                outfile.write(output+'\n')
            #self.scheduler.step(auc)
            if auc > best_score:
                best_score = auc
                torch.save(self.model, self.model_name)


    def eval(self, data_loader, form='mean'):
        self.model.eval()
        all_score = []
        all_y = []
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            score = self.model.predict(batch_x)
            all_score.append(score.data.cpu())
            all_y.append(batch_y.cpu())

        all_score = np.concatenate(all_score)
        all_y= np.concatenate(all_y)
            
        return evaluate(all_y, all_score, form)


