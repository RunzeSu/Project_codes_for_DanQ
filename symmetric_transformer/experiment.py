from model_transformer import *
import torch
import numpy as np
from torch.autograd import Variable
import os, time, argparse
import torch
from utils import *
import scipy.io as sio
import convtransLearner as CL
import seqlogo
from model_convtrans import  *
torch.cuda.empty_cache()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', default=False,
                         help='enable test.')
    parser.add_argument('--con', action='store_true', default=False,
                         help='enable continuing.')
    parser.add_argument('--onehot', action='store_true', default=False,
                         help='enable onehot.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--beta', type=float, default=0, help='Weight of graph laplacian')
    parser.add_argument('--hidden_size', type=int, default=12, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--load', action='store_true', default=False, help='renew data.')
    parser.add_argument('--time', action='store_true', default=False, help='Which task.')
    parser.add_argument('-P','--patience', type=int, default=30000, help='patience.')
    parser.add_argument('-F','--factor', type=float, default=0.95, help='factor.')
    parser.add_argument('-MT', '--model_type', type=str, default='conv', help='Model type.')
    parser.add_argument('-B','--batch_size', type=int, default=100, help='Batch size.')
    parser.add_argument('-NE','--nepochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('-L', '--learning_rate', type=float, default=0.0001, help='Initial learning rate.')
    parser.add_argument('-N', '--num_layer', type=int, default=2)
    parser.add_argument('-N_1', '--num_layer_1', type=int, default=2)
    parser.add_argument('-N_2', '--num_layer_2', type=int, default=1)
#    parser.add_argument('-N_3', '--num_layer_3', type=int, default=1)
    parser.add_argument('--local_size', type=int, default=3)
    parser.add_argument('--h', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=1024, help='dataset')
    parser.add_argument('--d_ff', type=int, default=1024, help='dataset')
    parser.add_argument('--data', type=str, default='ml', help='dataset')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    dropout = args.dropout
    batch_size = args.batch_size
    src_vocab = 5

    f_d = os.path.abspath(__file__)
    out_dir = os.sep.join(f_d.split(os.sep)[:-1]) + '/outputnow1/'

    onehot = True
    if onehot:
        dataset = './dataset/'
    
    train_loader, valid_loader, test_loader = load_data(args.batch_size, 
                                             is_onehot=onehot,
                                             is_shuffle=True,
                                             data_dir=dataset)
    
    """
    if args.cuda:
        train_x, train_y = train_x.cuda(), train_y.cuda()
        valid_x, valid_y = valid_x.cuda(), valid_y.cuda()
        test_x, test_y = test_x.cuda(), test_y.cuda()
    """

    name = '{}_ls_{}_ff_{}_dm_{}_N_{}_N1_{}_N2_{}_a_{}_b_{}_{}_weight{}_dropout{}'.format(
                            args.model_type, 
                            args.local_size,
                            args.d_ff,
                            args.d_model,
                            args.num_layer,
                            args.num_layer_1,
                            args.num_layer_2,
                            args.learning_rate,
                            args.batch_size,
                            args.nepochs,
                            args.wd,
                            args.dropout)


    model_name = out_dir+"m_{}".format(name)
    record_file = out_dir+"r_{}.txt".format(name)
    kwargs = {
	    'src_vocab':src_vocab,
	    'N': args.num_layer, 
	    'N_1': args.num_layer_1, 
	    'N_2': args.num_layer_2, 
	    'local_size': args.local_size, 
	    'd_model':args.d_model,
	    'd_ff':args.d_ff,
	    'h':args.h,
	    'n_class':919,
	    'dropout': args.dropout,
	    'model_name': model_name, 
	    'record_file': record_file,
	    'lr':args.learning_rate,
	    'bs':args.batch_size,
	    'test':args.test,
	    'wd':0.00000001,
        'con':args.con
      }
    print("model construction started...")
    learner = CL.convtransLearner(**kwargs)
    print("model construction finished!")
    #learner.model = torch.load('./JASPARmodel30float64')
    '''
    all_score = []    
    train_X=[]
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        score = learner.model.predictcnn(batch_x)
        all_score.append(score.data.cpu())
        train_X.append(batch_x.data.cpu())
        
    train_X = np.concatenate(train_X)   
    all_score = np.concatenate(all_score)
    print('print the predicted cnn')
    print(all_score.shape)
    '''
    #for i in range(8000):
 
    '''     
    def changekernels(length):
        kernelorder=sio.loadmat('orders.mat')['orders']
        learner_model = torch.load('./output/m_conv_ls_3_ff_1024_dm_1024_N_2_N1_2_N2_1_a_0.0001_b_100_200_weight0.0005_dropout0.1')
        a=kernelorder[0:length-1].T[0]
        #newweight=learner_model.state_dict()['conv.weight'][a.astype(np.int16)]
        learner.model.conv.weight=torch.nn.Parameter(learner_model.state_dict()['conv.weight'][a.astype(np.int16)])
        
    changekernels(400)
    '''

    #sio.savemat('pred_cnn.mat', {'predcnn': all_score})
    ###########################################################################
    if not args.test:
        learner.train(args.nepochs, train_loader, valid_loader, test_loader)
        AUC = learner.eval(test_loader, 'all')
        s =  'Test_result: \n{}'.format(AUC)
        with open(record_file, 'a') as outfile:
            outfile.write(s)
        print (s)
    else:
        record_file = out_dir+"test_{}".format(name)
        AUC = learner.eval(test_loader, 'all')
        s =  'Test_result: \n{}'.format(AUC)
        with open(record_file, 'w') as outfile:
            outfile.write(s)
        print (s)

if __name__ == "__main__":
    main()  


