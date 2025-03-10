#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="total number of clients: N")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--schedule', type=int, nargs='*', default=[162, 244],
                        help='Decrease learning rate at these rounds.')
    parser.add_argument('--lr_decay',type = float,default=0.1,
                        help = 'Learning rate decay at specified rounds')
    parser.add_argument('--momentum', type=float, default=0.0,
                        help='SGD momentum (default: 0.0)')
    parser.add_argument('--reg', default=1e-4, type=float, 
                        help='weight decay for an optimizer')
    parser.add_argument('--global_average',action = 'store_true',
                        help='use all clients (including which are not updated in this round) for averaging')
    parser.add_argument('--FedProx',action='store_true',
                        help='use FedProx')
    parser.add_argument('--mu',type = float, default=0.0,
                        help = 'mu in FedProx')
    parser.add_argument('--dynamic_mu',action = 'store_true',
                        help='use a dynamic mu for FedProx')


    # Power-d arguments
    parser.add_argument('--power_d',action = 'store_true',
                        help = 'use Pow-d selection')
    parser.add_argument('--d',type = int,default = 30,
                        help='d in Pow-d selection')

    # Active Federated Learning arguments
    parser.add_argument('--afl',action = 'store_true',
                        help = 'use AFL selection')
    parser.add_argument('--alpha1',type = float,default=0.75,
                        help = 'alpha_1 in ALF')
    parser.add_argument('--alpha2',type = float,default=0.01,
                        help = 'alpha_2 in AFL')
    parser.add_argument('--alpha3',type = float,default=0.1,
                        help='alpha_3 in AFL')

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', 
                        help='model name')
    parser.add_argument('--kernel_sizes', type=int, default=[5,5],nargs="*",
                        help='kernel size in each convolutional layer')
    parser.add_argument('--num_filters', type=int, default=[32,64],nargs = "*",
                        help="number of filters in each convolutional layer.")
    parser.add_argument('--padding', action='store_true', 
                        help='use padding in each convolutional layer')
    parser.add_argument('--mlp_layers',type= int,default=[64,],nargs="*",
                        help="numbers of dimensions of each hidden layer in MLP, or fc layers in CNN")
    parser.add_argument('--depth',type = int,default = 20, 
                        help = "The depth of ResNet. Only valid when model is resnet")
    

    # utils arguments
    parser.add_argument('--dataset', type=str, default='mnist', 
                        help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, 
                        help="number of classes")
    parser.add_argument('--gpu', default=None, 
                        help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', 
                        help="type of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--shards_per_client',type = int,default=1,
                        help='number of shards for each client')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unbalanced data splits for non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--alpha',type=float,default=None,
                        help="use Dirichlet_noniid sampling, set the alpha of Dir here")
    parser.add_argument('--verbose', type=int, default=1, 
                        help='verbose')
    parser.add_argument('--seed', type=int, default=None, nargs='*', 
                        help='random seed')
    parser.add_argument('--target_accuracy',type=float,default=None,
                        help='stop at a specified test accuracy')
  
    
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = args_parser()
    print(args.mlp_layers)
