#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch

from options import args_parser
from update import LocalUpdate,test_inference,train_federated_learning,federated_test_idx
from models import MLP, NaiveCNN, BNCNN, ResNet,RNN
from utils import get_dataset, average_weights, exp_details,setup_seed


if __name__ == '__main__':
    os.environ["OUTDATED_IGNORE"]='1'
    start_time = time.time()
    # define paths
    path_project = os.path.abspath('..')

    args = args_parser()
    gargs = copy.deepcopy(args)
    exp_details(args)
    if not args.iid:
        base_file = './save/objects/Pow-d_files/(Pow-d25){}_{}_{}_{}_C[{}]_iid[{}]_{}[{}]_E[{}]_B[{}]_mu[{}]_lr[{:.5f}]'.\
                    format(args.dataset,'FedProx[%.3f]'%args.mu if args.FedProx else 'FedAvg', args.model, args.epochs,args.frac, args.iid,
                    'sp' if args.alpha is None else 'alpha',args.shards_per_client if args.alpha is None else args.alpha,
                    args.local_ep, args.local_bs,args.mu,args.lr)
    else:
        base_file = './save/objects/Pow-d_files/(Pow-d25){}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_mu[{}]_lr[{:.5f}]'.\
                    format(args.dataset,'FedProx[%.3f]'%args.mu if args.FedProx else 'FedAvg', args.model, args.epochs,args.frac, args.iid,
                    args.local_ep, args.local_bs,args.mu,args.lr)

    if not os.path.exists(base_file):
        os.makedirs(base_file)
    
    if args.afl:
        file_name = base_file+'/afl'
    elif args.power_d:
        file_name = base_file+'/powerd_d[{}]'.format(args.d)
    else:
        file_name = base_file+'/random'

    device = 'cuda:'+args.gpu if args.gpu else 'cpu'
    if args.gpu:
        torch.cuda.set_device(device)
    if gargs.seed is None or gargs.iid:
        gargs.seed = [None,]

    time_list = []  # record the running time for each seed 
    for seed in gargs.seed:
        args = copy.deepcopy(gargs)# recover the args
        print("Start with Random Seed: {}".format(seed))
        # load dataset and user groups
        train_dataset, test_dataset, user_groups, user_groups_test,weights = get_dataset(args,seed)
        # weights /=np.sum(weights)
        if seed is not None:
            setup_seed(seed)
        data_size = train_dataset[0][0].shape
        # BUILD MODEL
        if args.model == 'cnn':
            # Naive Convolutional neural netork
            global_model = NaiveCNN(args=args,input_shape = data_size,final_pool=False)
        
        elif args.model == 'bncnn':
            # Convolutional neural network with batch normalization
            global_model = BNCNN(args = args, input_shape = data_size)

        elif args.model == 'mlp' or args.model == 'log':
            # Multi-layer preceptron
            len_in = 1
            for x in data_size:
                len_in *= x
                global_model = MLP(dim_in=len_in, dim_hidden=args.mlp_layers if args.model=='mlp' else [],
                                dim_out=args.num_classes)
        elif args.model == 'resnet':
            global_model = ResNet(args.depth,args.num_classes)
        elif args.model == 'rnn':
            if args.dataset=='shake':
                global_model = RNN(256,args.num_classes)
            else:
                # emb_arr,_,_= get_word_emb_arr('./data/sent140/embs.json')
                global_model = RNN(256,args.num_classes,300,True,128)
        else:
            exit('Error: unrecognized model')

        # Set the model to train and send it to device.
        global_model.to(device)
        global_model.train()
        print(global_model)

        # copy weights
        global_weights = global_model.state_dict()
        local_weights = []# store local weights of all users for averaging
        local_states = []# store local states of all users, these parameters should not be uploaded

        
        for i in range(args.num_users):
            local_states.append(copy.deepcopy(global_model.Get_Local_State_Dict()))
            local_weights.append(copy.deepcopy(global_weights))

        local_states = np.array(local_states)
        local_weights = np.array(local_weights)

        # Training
        train_loss, train_accuracy = [], []
        test_loss,test_accuracy = [],[]
        test_losses = []
        max_accuracy=0.0

        local_losses = []# test losses evaluated on local models(before averaging)
        chosen_clients = []# chosen clients on each epoch
        gt_global_losses = []# test losses on global models(after averaging) over all clients
 
        print_every = 1
        init_mu = args.mu

        predict_losses = []

        # sigma = []
        # sigma_gt=[]

        #init parameters
        AFL_Valuation_list = []
        # prob_list = []
        Theta = 0.7
        Theta_w = 0.99995
        win_theta = 0
        # Beta = 0.5
        # Rho = 0.7
        # prob_temp = 0
        # temp1,temp2=(0,0)
        # gt_global_losses_list = []

        # mu_p_list = []
        # sigma_p_list = []

        # Test the global model before training
        list_acc, list_loss = federated_test_idx(args,global_model,
                                                list(range(args.num_users)),
                                                train_dataset,user_groups)
        gt_global_losses.append(list_loss)

        gt_global_losses_temp = gt_global_losses
        # print('---------------------gt_global_losses_befortraining:',gt_global_losses)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        #      
        if args.afl:
            # print('!!!!!!!!!!!!!!!!!!list_loss', len(list_loss),list_loss)
            AFL_Valuation = np.array(list_loss)*np.sqrt(weights*len(train_dataset))
            AFL_Valuation_temp = AFL_Valuation     #每一seed中AFL_Valuation_temp初始值为第一个AFL_Valuation

        m = max(int(args.frac * args.num_users), 1)
        for epoch in tqdm(range(args.epochs)):
            print('\n | Global Training Round : {} |\n'.format(epoch+1))
            epoch_global_losses = []
            epoch_local_losses = []
            global_model.train()
            # gt_global_losses = []     #Doris:pow-d
            if args.dataset=='cifar' or epoch in args.schedule:
                args.lr*=args.lr_decay

            #Doris： 迭代递增的客户端选择算法
            # m = max(int(args.frac * args.num_users), 1)
            if (epoch % 11 == 0 and epoch != 0):        # and epoch <= 100
                m = m + 1
            # if epoch >= 100:
            #     m = max(int(args.frac * args.num_users), 1)
            print('***************************the number of clients:',m)

            
            if args.afl:
                # AFL
                #Doris:1     把后面的list_loss传入AFL_Valuation计算，而不是每一个seed只计算一次
                AFL_Valuation = np.array(list_loss) * np.sqrt(weights * len(train_dataset))
                AFL_Valuation_list.append(AFL_Valuation)
                AFL_Valuation_temp = Theta * AFL_Valuation_list[epoch] + (1-Theta)*AFL_Valuation_temp     #在vk计算中加入衰减系数theta（AFL-2的公式）
                win_theta = pow(Theta_w,epoch)
                # AFL_Valuation_temp = pow(Theta,win_theta) * AFL_Valuation_list[epoch] + (1 - pow(Theta,win_theta)) * AFL_Valuation_temp     #AFL-36的算法
                # AFL_Valuation_temp = (Theta / win_theta) * AFL_Valuation_list[epoch] + (1 - Theta / win_theta) * AFL_Valuation_temp       #AFL-44的算法
                # AFL_Valuation_temp = AFL_Valuation_list[epoch]-AFL_Valuation_list[epoch-1]      #Doris:vk和本轮和上一轮的损失差成正比
                # print('!!!!!!!!!!!!!!!!!!!!!!!!AFL_Valuation_list,epoch:',AFL_Valuation_list,epoch)
                #vk计算中加入AFL-25公式
                # for i in range(epoch+1):
                #     # print('!!!!!!!!!!!!!!!!!!!!!!!!AFL_Valuation_list[epoch]:',AFL_Valuation_list[epoch])
                #     temp1 += pow(Rho,epoch-i)*AFL_Valuation_list[i]
                #     temp2 += pow(Rho,epoch-i)
                # AFL_Valuation_temp = temp1

                delete_num = int(args.alpha1*args.num_users)    #delete_num = α1*K
                sel_num = int((1-args.alpha3)*m)        #sel_num = (1-α3)m
                #Doris:2
                tmp_value = np.vstack([np.arange(args.num_users),AFL_Valuation_temp])   #把AFL_Valuation换成了AFL_Valuation_tmp

                # tmp_value = np.vstack([np.arange(args.num_users),AFL_Valuation])
                tmp_value = tmp_value[:,tmp_value[1,:].argsort()]   #根据Vk进行排序

                #Doris：选择概率改进
                # prob = np.exp(args.alpha2 * tmp_value[1, delete_num:])      #在pk处加入迭代运算
                # prob_list.append(prob)
                # if epoch == 0:
                #     prob_temp = prob
                # else:
                #     prob_temp = Beta * prob_list[epoch] + (1-Beta) * prob_temp
                # prob = prob_temp/np.sum(prob_temp)

                #Doris:将选择概率写成对数形式
                # prob = np.log(tmp_value[1,delete_num:])
                # prob = prob/np.sum(prob)

                prob = np.exp(args.alpha2*tmp_value[1,delete_num:])
                prob = prob/np.sum(prob)            #选择概率pk(将最小的置-∞后剩25个，α1=0.75)
                sel1 = np.random.choice(np.array(tmp_value[0,delete_num:],dtype=np.int64),sel_num,replace=False,p=prob) #集合S‘
                remain = set(np.arange(args.num_users))-set(sel1)       #剩余集合
                sel2 = np.random.choice(list(remain),m-sel_num,replace = False)     #集合S’‘
                idxs_users = np.append(sel1,sel2)   #合并S’和S‘’(5个)
                # print('-------------------idxs_users',len(idxs_users),idxs_users)

                # AFL：源码
                # delete_num = int(args.alpha1*args.num_users)
                # sel_num = int((1-args.alpha3)*m)
                # tmp_value = np.vstack([np.arange(args.num_users),AFL_Valuation])
                # tmp_value = tmp_value[:,tmp_value[1,:].argsort()]
                # prob = np.exp(args.alpha2*tmp_value[1,delete_num:])
                # prob = prob/np.sum(prob)
                # sel1 = np.random.choice(np.array(tmp_value[0,delete_num:],dtype=np.int64),sel_num,replace=False,p=prob)
                # remain = set(np.arange(args.num_users))-set(sel1)
                # sel2 = np.random.choice(list(remain),m-sel_num,replace = False)
                # idxs_users = np.append(sel1,sel2)

            elif args.power_d:
                # Power-of-D-choice
                # Test the global model before training
                #Doris:pow-d  在每一轮更新损失值
                # list_acc, list_loss = federated_test_idx(args, global_model,
                #                                          list(range(args.num_users)),
                #                                          train_dataset, user_groups)
                # gt_global_losses.append(list_loss)
                # gt_global_losses_list.append(gt_global_losses[0])
                # win_theta = pow(Theta_w, epoch)
                # gt_global_losses_temp = [[Theta * gt_global_losses_list[epoch][i] + (1 - Theta ) * gt_global_losses_temp[0][i] for i in range(len(gt_global_losses_temp[0]))]]
                # gt_global_losses_temp = [[(Theta / win_theta)*gt_global_losses_list[epoch][i] + (1-Theta / win_theta)*gt_global_losses_temp[0][i] for i in range(len(gt_global_losses_temp[0]))]]     #在pk计算中gt_global_losses加入衰减系数theta（AFL-2、pow-d3的公式）

                A = np.random.choice(range(args.num_users), args.d, replace=False,p=weights)
                # idxs_users = A[np.argsort(np.array(gt_global_losses_temp[-1])[A])[-m:]]         #Doris

                idxs_users = A[np.argsort(np.array(gt_global_losses[-1])[A])[-m:]]

            else:
                # Random selection
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)

            chosen_clients.append(idxs_users)
            # print('-----------------------------chosen_clients',len(chosen_clients),chosen_clients)
            
            for idx in idxs_users:
                local_model = copy.deepcopy(global_model)
                local_update = LocalUpdate(args=args, dataset=train_dataset,
                                        idxs=user_groups[idx] ,global_round = epoch)
                w,test_loss,init_test_loss = local_update.update_weights(model=local_model)
                
                local_states[idx] = copy.deepcopy(local_model.Get_Local_State_Dict())
                local_weights[idx]=copy.deepcopy(w)
                epoch_global_losses.append(init_test_loss)# TAKE CARE: this is the test loss evaluated on the (t-1)-th global weights!
                epoch_local_losses.append(test_loss)
                # print('!!!!!!!!!!!!!!!!!!!!!epoch_local_loss',len(epoch_local_losses),epoch_local_losses)

            # update global weights
            if args.global_average:
                global_weights = average_weights(local_weights,omega=None)
            else:
                global_weights = average_weights(local_weights[idxs_users],omega=None)

            for i in range(args.num_users):
                local_weights[i] = copy.deepcopy(global_weights)
            # update global weights
            global_model.load_state_dict(global_weights)

            if args.afl:
                AFL_Valuation[idxs_users] = np.array(epoch_global_losses)*np.sqrt(weights[idxs_users]*len(train_dataset))
            local_losses.append(epoch_local_losses)

            # dynamic mu for FedProx
            loss_avg = sum(epoch_local_losses) / len(epoch_local_losses)
            if args.dynamic_mu and epoch>0:
                if loss_avg>loss_prev:
                    args.mu+=init_mu*0.1
                else:
                    args.mu=max([args.mu-init_mu*0.1,0.0])
            loss_prev = loss_avg
            train_loss.append(loss_avg)

            # calculate test accuracy over all users
            list_acc, list_loss = federated_test_idx(args,global_model,
                                                    list(range(args.num_users)),
                                                    train_dataset,user_groups)
            # print('$$$$$$$$$$$$$$$$$$$$$$$$$$list_loss_after',len(list_loss),list_loss)
            gt_global_losses.append(list_loss)
            train_accuracy.append(sum(list_acc)/len(list_acc))



            # test inference on the global test dataset
            test_acc, test_loss = test_inference(args, global_model, test_dataset)
            test_accuracy.append(test_acc)
            test_losses.append(test_loss)
            if args.target_accuracy is not None:
                if test_acc>=args.target_accuracy:
                    break


            # print global training loss after every 'i' rounds
            if (epoch+1) % print_every == 0:
                print(' \nAvg Training Stats after {} global rounds:'.format(epoch+1))
                print('Training Loss : {}'.format(np.sum(np.array(list_loss)*weights)))
                print("Test Accuracy: {:.2f}%\n".format(100*test_acc))

        # write test_accuracy in tensorboard
        writer_acc_path = SummaryWriter(base_file+'/tensorboard_files/{}_{}—{}'.format('acc_seed',seed,time.strftime('%Y-%m-%dT%H-%M-%S')))
        for index, data in enumerate(test_accuracy):
            writer_acc_path.add_scalar('test_accuracy', test_accuracy[index], index)
        writer_acc_path.close()

        # write test_loss in tensorboard
        writer_loss_path = SummaryWriter(
            base_file + '/tensorboard_files/{}_{}—{}'.format('loss_seed', seed, time.strftime('%Y-%m-%dT%H-%M-%S')))
        for index, data in enumerate(test_losses):
            writer_loss_path.add_scalar('test_losses', test_losses[index], index)
        writer_loss_path.close()
        
        print(' \n Results after {} global rounds of training:'.format(epoch+1))
        print("|---- Final Test Accuracy: {:.2f}%".format(100*test_accuracy[-1]))
        print("|---- Max Test Accuracy: {:.2f}%".format(100*max(test_accuracy)))
        if args.gpr:
            print("|---- Mean GP Prediction Loss: {:.4f}".format(np.mean(predict_losses)))

        print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
        runtime = time.time()-start_time
        time_list.append(runtime)
        print('Time list in seeds:',time_list)

        # save the training records:
        with open(file_name+'_{}.pkl'.format(seed), 'wb') as f:
            pickle.dump([train_loss, train_accuracy,chosen_clients,
                        weights, None,
                        gt_global_losses,test_accuracy], f)
        

        
        
        
