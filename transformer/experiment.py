from model_transformer import *
import torch
import numpy as np
from torch.autograd import Variable
import os, time, argparse
import torch
from utils import *
import modelLearner as ML
import xtransLearner as XL
import convtransLearner as CL


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
    parser.add_argument('-NE','--nepochs', type=int, default=60, help='Number of epochs to train.')
    parser.add_argument('-L', '--learning_rate', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('-N', '--num_layer', type=int, default=2)
    parser.add_argument('-N_1', '--num_layer_1', type=int, default=2)
    parser.add_argument('-N_2', '--num_layer_2', type=int, default=1)
    parser.add_argument('--local_size', type=int, default=3)
    parser.add_argument('--h', type=int, default=4)
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
    out_dir = os.sep.join(f_d.split(os.sep)[:-1]) + '/output/'
    print(out_dir)

    onehot = False
    if args.model_type == 'conv':
        onehot = True
    
    if onehot:
        dataset = '/mnt/ufs18/home-052/surunze/biostat_project/archive_1/transcheckkernels1200/dataset/'
    else:
        dataset = '/mnt/ufs18/home-052/surunze/biostat_project/archive_1/transcheckkernels1200/dataset/'
    print(dataset)
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

    if args.model_type == 'xtrans':
        name = '{}_ls_{}_ff_{}_dm_{}_N1_{}_N2_{}_a_{}'.format(
                            args.model_type, 
                            args.local_size,
                            args.d_ff,
                            args.d_model,
                            args.num_layer_1,
                            args.num_layer_2,
                            args.learning_rate)
    else:
        name = '{}_ff_{}_dm_{}_N_{}_a_{}_drop{}'.format(
                            args.model_type, 
                            args.d_ff,
                            args.d_model,
                            args.num_layer,
                            args.learning_rate,
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
	    'wd':0.001,
        'con':args.con
	    }

    print("learner starting")
    if args.model_type == 'xtrans':
        learner = XL.xtransLearner(**kwargs)
    elif args.model_type == 'conv':
        learner = CL.convtransLearner(**kwargs)
    else:
        learner = ML.modelLearner(**kwargs)
    print("learner constructed!")
    ###########################################################################
    if not args.test:
        learner.train(args.nepochs, train_loader, valid_loader)
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


