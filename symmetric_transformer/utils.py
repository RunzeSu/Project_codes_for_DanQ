import torch
import numpy as np
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_curve
from torch.utils import data
import h5py
import scipy.io

class Dataset(data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
        
        
def pr_auc_score(labels, scores):
    precision, recall, th = precision_recall_curve(labels, scores)
    auc1=auc( recall, precision)
    return auc1
        
        

def load_data(batch_size=64, is_onehot=False, is_shuffle=True, data_dir="../dataset/"):
    """
	# X_test = torch.FloatTensor(np.load(data_dir + 'X_test.npy'))	
	# Y_test = torch.FloatTensor(np.load(data_dir + 'Y_test.npy'))
	X_valid = torch.LongTensor(np.load(data_dir + 'X_valid.npy'))
	X_valid = X_valid.view(X_valid.shape[0], -1)
	Special_valid = torch.LongTensor(np.ones((X_valid.shape[0],1))) * 4

	X_valid = torch.cat((Special_valid, X_valid), dim=1)
	#X_valid = X_valid[:200]
	Y_valid = torch.FloatTensor(np.load(data_dir + 'Y_valid.npy'))
	#Y_valid = Y_valid[:200]
	return X_valid, Y_valid, X_valid, Y_valid
    """
    para = {'batch_size':batch_size,
            'shuffle': is_shuffle}

    if is_onehot:
        print ('load train')
        
        train_x = torch.FloatTensor(np.transpose(np.array((h5py.File('dataset/train.mat', 'r'))['trainxdata']),axes=(2,0,1)))[:2200000]
        train_y = torch.FloatTensor(np.array((h5py.File('dataset/train.mat', 'r'))['traindata']).T)[:2200000]
        #train_x = torch.FloatTensor(np.transpose(scipy.io.loadmat('dataset/train.mat')['trainxdata'],axes=(0,2,1)))
        #train_y = torch.FloatTensor((scipy.io.loadmat('dataset/train.mat'))['traindata'])
        para['batch_size'] = batch_size
        train_set = Dataset(train_x, train_y)
        train_loader = data.DataLoader(train_set, **para)
        """
        train_x = torch.FloatTensor(np.transpose(scipy.io.loadmat('dataset/test.mat')['testxdata'],axes=(0,2,1)))
        train_y = torch.FloatTensor((scipy.io.loadmat('dataset/test.mat'))['testdata'])

        train_set = Dataset(train_x, train_y)
        train_loader = data.DataLoader(train_set, **para)
        """
        print ('load test')
        test_x = torch.FloatTensor(np.transpose(scipy.io.loadmat('dataset/test.mat')['testxdata'],axes=(0,2,1)))
        test_y = torch.FloatTensor((scipy.io.loadmat('dataset/test.mat'))['testdata'])

        test_set = Dataset(test_x, test_y)
        test_loader = data.DataLoader(test_set, **para)


        print ('load valid')
        valid_x = torch.FloatTensor(np.transpose(scipy.io.loadmat('dataset/valid.mat')['validxdata'],axes=(0,2,1)))
        valid_y = torch.FloatTensor((scipy.io.loadmat('dataset/valid.mat'))['validdata'])

        valid_set = Dataset(valid_x, valid_y)
        valid_loader = data.DataLoader(valid_set, **para)


    else:
        train_x = torch.LongTensor(np.load(data_dir + 'train_x.npy'))
        train_x = train_x.view(train_x.shape[0], -1)
        #train_x = train_x[:, :500]
        #train_x = train_x[:500]
        train_special = torch.LongTensor(np.ones((train_x.shape[0],1))) * 4
        train_x = torch.cat((train_special, train_x), dim=1)
        train_y = torch.FloatTensor(np.load(data_dir + 'train_y.npy'))

        para['batch_size'] = batch_size
        train_set = Dataset(train_x, train_y)
        train_loader = data.DataLoader(train_set, **para)

        para['batch_size'] = batch_size//3
        test_x = torch.LongTensor(np.load(data_dir + 'test_x.npy'))
        test_x = test_x.view(test_x.shape[0], -1)
        test_special = torch.LongTensor(np.ones((test_x.shape[0],1))) * 4
        test_x = torch.cat((test_special, test_x), dim=1)
        test_y = torch.FloatTensor(np.load(data_dir + 'test_y.npy'))
        test_set = Dataset(test_x, test_y)
        test_loader = data.DataLoader(test_set, **para)

        para['batch_size'] = batch_size//4
        valid_x = torch.LongTensor(np.load(data_dir + 'valid_x.npy'))
        valid_x = valid_x.view(valid_x.shape[0], -1)
        valid_special = torch.LongTensor(np.ones((valid_x.shape[0],1))) * 4
        valid_x = torch.cat((valid_special, valid_x), dim=1)
        valid_y = torch.FloatTensor(np.load(data_dir + 'valid_y.npy'))

        valid_set = Dataset(valid_x, valid_y)
        valid_loader = data.DataLoader(valid_set, **para)

    print (train_x.shape, valid_x.shape, test_x.shape)
    return  train_loader, valid_loader, test_loader




def evaluate(labels, scores, form):
    if form == 'mean':
        auc_list = []
        m_list = []
        n_class = labels.shape[1]
        for i in range(n_class):
            if not np.count_nonzero(labels[:,i]) == 0:
                auc1 = pr_auc_score(labels[:,i], scores[:,i])
                auc_list.append('{:.4f}'.format(auc1))
                m_list.append(auc1)
        auc1=sum(m_list)/918
        return auc1
    else:
        auc_list = []
        m_list = []
        n_class = labels.shape[1]
        for i in range(n_class):
            if not np.count_nonzero(labels[:,i]) == 0:
                auc1 = pr_auc_score(labels[:,i], scores[:,i])
                auc_list.append('{:.4f}'.format(auc1))
                m_list.append(auc1)
        #return ",".join(auc_list) + "\n{}\nMean: {:.4f}".format(len(m_list),np.mean(m_list))
        return(auc_list)









def evaluatepositive(labels, scores, form):
    if form == 'mean':
        auc_list = []
        m_list = []
        n_class = labels.shape[1]
        for i in range(n_class):
            if not np.count_nonzero(labels[:,i]) == 0:
                auc1 = pr_auc_score(labels[0:227512,i], scores[0:227512,i])
                auc_list.append('{:.4f}'.format(auc1))
                m_list.append(auc1)
        auc1=sum(m_list)/918
        return auc1
    else:
        auc_list = []
        m_list = []
        n_class = labels.shape[1]
        for i in range(n_class):
            if not np.count_nonzero(labels[:,i]) == 0:
                auc1 = pr_auc_score(labels[0:227512,i], scores[0:227512,i])
                auc_list.append('{:.4f}'.format(auc1))
                m_list.append(auc1)
        #return ",".join(auc_list) + "\n{}\nMean: {:.4f}".format(len(m_list),np.mean(m_list))
        return(auc_list)






def evaluatepositiveroc(labels, scores, form):
    if form == 'mean':
        auc_list = []
        m_list = []
        n_class = labels.shape[1]
        for i in range(n_class):
            if not np.count_nonzero(labels[:,i]) == 0:
                auc1 = roc_auc_score(labels[0:227512,i], scores[0:227512,i])
                auc_list.append('{:.4f}'.format(auc1))
                m_list.append(auc1)
        auc1=sum(m_list)/918
        return auc1
    else:
        auc_list = []
        m_list = []
        n_class = labels.shape[1]
        for i in range(n_class):
            if not np.count_nonzero(labels[:,i]) == 0:
                auc1 = roc_auc_score(labels[0:227512,i], scores[0:227512,i])
                auc_list.append('{:.4f}'.format(auc1))
                m_list.append(auc1)
        #return ",".join(auc_list) + "\n{}\nMean: {:.4f}".format(len(m_list),np.mean(m_list))
        return(auc_list)




def evaluatereverse(labels, scores, form):
    if form == 'mean':
        auc_list = []
        m_list = []
        n_class = labels.shape[1]
        for i in range(n_class):
            if not np.count_nonzero(labels[:,i]) == 0:
                auc1 = pr_auc_score(labels[227512:455024,i], scores[227512:455024:i])
                auc_list.append('{:.4f}'.format(auc1))
                m_list.append(auc1)
        auc1=sum(m_list)/918
        return auc1
    else:
        auc_list = []
        m_list = []
        n_class = labels.shape[1]
        for i in range(n_class):
            if not np.count_nonzero(labels[:,i]) == 0:
                auc1 = pr_auc_score(labels[227512:455024,i], scores[227512:455024,i])
                auc_list.append('{:.4f}'.format(auc1))
                m_list.append(auc1)
        #return ",".join(auc_list) + "\n{}\nMean: {:.4f}".format(len(m_list),np.mean(m_list))
        return(auc_list)






def evaluatereverseroc(labels, scores, form):
    if form == 'mean':
        auc_list = []
        m_list = []
        n_class = labels.shape[1]
        for i in range(n_class):
            if not np.count_nonzero(labels[:,i]) == 0:
                auc1 = roc_auc_score(labels[227512:455024,i], scores[227512:455024,i])
                auc_list.append('{:.4f}'.format(auc1))
                m_list.append(auc1)
        auc1=sum(m_list)/918
        return auc1
    else:
        auc_list = []
        m_list = []
        n_class = labels.shape[1]
        for i in range(n_class):
            if not np.count_nonzero(labels[:,i]) == 0:
                auc1 = roc_auc_score(labels[227512:455024,i], scores[227512:455024,i])
                auc_list.append('{:.4f}'.format(auc1))
                m_list.append(auc1)
        #return ",".join(auc_list) + "\n{}\nMean: {:.4f}".format(len(m_list),np.mean(m_list))
        return(auc_list)










def evaluateroc(labels, scores, form):
    if form == 'mean':
        auc_list = []
        m_list = []
        n_class = labels.shape[1]
        for i in range(n_class):
            if not np.count_nonzero(labels[:,i]) == 0:
                auc1 = roc_auc_score(labels[:,i], scores[:,i])
                auc_list.append('{:.4f}'.format(auc1))
                m_list.append(auc1)
        auc1=sum(m_list)/918
        return auc1
    else:
        auc_list = []
        m_list = []
        n_class = labels.shape[1]
        for i in range(n_class):
            if not np.count_nonzero(labels[:,i]) == 0:
                auc1 = roc_auc_score(labels[:,i], scores[:,i])
                auc_list.append('{:.4f}'.format(auc1))
                m_list.append(auc1)
        #return ",".join(auc_list) + "\n{}\nMean: {:.4f}".format(len(m_list),np.mean(m_list))
        return(auc_list)