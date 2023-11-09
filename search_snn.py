import os
import time
import torch
import utils
import config
import torchvision
import torch.nn as nn
import numpy as np
import random
from model_snn import Supernet, is_single_path, prune_func_rank
from utils import data_transforms
from spikingjelly.clock_driven.functional import reset_net
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import torch.nn.functional as F

def main():
    args = config.get_args()
    train_transform, valid_transform = data_transforms(args)
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar10'), train=True,
                                                download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, num_workers=4)
        valset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar10'), train=False,
                                              download=True, transform=valid_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=4)
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=os.path.join(args.data_dir, 'cifar100'), train=True,
                                                download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, num_workers=4)
        valset = torchvision.datasets.CIFAR100(root=os.path.join(args.data_dir, 'cifar100'), train=False,
                                              download=True, transform=valid_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=4)
    elif args.dataset == 'tinyimagenet':
        trainset = torchvision.datasets.ImageFolder(os.path.join(args.datadir,'/tiny-imagenet-200/train'),
                                        train_transform)
        valset = torchvision.datasets.ImageFolder(os.path.join(args.datadir,'/tiny-imagenet-200/val'),
                                      valid_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=4, pin_memory=True, sampler=None)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=4, pin_memory=True)

    elif args.dataset == 'DVS128Gesture':
      trainset = DVS128Gesture(root=args.datadir, train=True, data_type='frame', frames_number=args.timestep, split_by='number')
    
    start = time.time()
    search_space = args.search_space
    network = Supernet(args,max_nodes=4, search_space=search_space)
    arch_parameters = [alpha.detach().clone() for alpha in network.get_alphas()]
    for alpha in arch_parameters:
        alpha[:, :] = 0
    
    network = network.cuda()

    INF = 1000
    arch_parameters_history = []
    arch_parameters_history_npy = []
    start_time = time.time()
    epoch = -1

    #arch_parameters_history.append([alpha.detach().clone() for alpha in arch_parameters])
    #arch_parameters_history_npy.append([alpha.detach().clone().cpu().numpy() for alpha in arch_parameters])
    #np.save(os.path.join('/content/drive/MyDrive/', "arch_parameters_history.npy"), arch_parameters_history_npy)
    
    while not is_single_path(network):
        epoch += 1
        print('epoch:',epoch)
        torch.cuda.empty_cache()

        arch_parameters, op_pruned = prune_func_rank(args, arch_parameters, trainset, search_space,network)
        network.set_alphas(arch_parameters)

        #arch_parameters_history.append([alpha.detach().clone() for alpha in arch_parameters])
        #arch_parameters_history_npy.append([alpha.detach().clone().cpu().numpy() for alpha in arch_parameters])
        #np.save(os.path.join('/content/drive/MyDrive/', "arch_parameters_history.npy"), arch_parameters_history_npy)


    end_time = time.time()
    
    print ('-'*7, "best_neuroncell",'-'*7)
    print (arch_parameters)
    print('-' * 30)
    utils.time_record(start)

    # Reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    search_space = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    model = Supernet(args,max_nodes=4, search_space=search_space).cuda()
    model.set_alphas(arch_parameters[0])
    criterion = nn.CrossEntropyLoss().cuda()


    if args.savemodel_pth is not None:
        print (torch.load(args.savemodel_pth).keys())
        model.load_state_dict(torch.load(args.savemodel_pth)['state_dict'])
        print ('test only...')
        validate(args, 0, val_loader, model, criterion)
        exit()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
    elif args.optimizer == 'adam':
          optimizer = torch.optim.Adam(model.parameters(),  args.learning_rate,args.weight_decay)
    else:
        print ("will be added...")
        exit()

    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.5),int(args.epochs*0.75)], gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= int(args.epochs), eta_min= args.learning_rate*0.01)
    else:
        print ("will be added...")
        exit()



    #start = time.time()
    #for epoch in range(args.epochs):
    #    print('epoch number:',epoch)
    #    train(args, epoch, train_loader, model, criterion, optimizer, scheduler)
    #    scheduler.step()
    #    #if (epoch + 1) % args['val_interval'] == 0:
    #    validate(args, epoch, val_loader, model, criterion)
          #save_checkpoint({'state_dict': model.state_dict(), }, epoch + 1, tag=args['exp_name'] + '_super')
    #utils.time_record(start)


def train(args, epoch, train_data,  model, criterion, optimizer, scheduler):
    model.train()
    train_loss = 0.0
    top1 = utils.AvgrageMeter()
    if (epoch + 1) % 10 == 0:
        print('[%s%04d/%04d %s%f]' % ('Epoch:', epoch + 1, args.epochs, 'lr:', scheduler.get_lr()[0]))

    for step, (inputs, targets) in enumerate(train_data):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        if args.dataset =='DVS128Gesture':
            inputs = inputs.permute(1, 0, 2, 3, 4)
            label_onehot = F.one_hot(targets, 11).float()
        outputs = model(inputs)
        if args.dataset=='DVS128Gesture':
            loss = F.mse_loss(outputs, label_onehot)
        else:
            loss = criterion(outputs, targets)
        #loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
        if args.dataset=='DVS128Gesture':
            n = inputs.size(1)
        else:
            n = inputs.size(0)
        #n = inputs.size(0)
        top1.update(prec1.item(), n)
        train_loss += loss.item()
        reset_net(model)
    print('train_loss: %.6f' % (train_loss / len(train_data)), 'train_acc: %.6f' % top1.avg)


def validate(args, epoch, val_data, model, criterion):
    model.eval()
    val_loss = 0.0
    val_top1 = utils.AvgrageMeter()

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_data):
            if args.dataset=='DVS128Gesture':
                inputs = inputs.permute(1, 0, 2, 3, 4)
                label_onehot = F.one_hot(targets, 11).float()

            outputs = model(inputs)
            if args.dataset=='DVS128Gesture':
                loss = F.mse_loss(outputs, label_onehot)
            else:
                loss = criterion(outputs, targets)
            val_loss += loss.item()
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            if args.dataset=='DVS128Gesture':
                n = inputs.size(1)
            else:
                n = inputs.size(0)
            val_top1.update(prec1.item(), n)
            reset_net(model)
        print('[Val_Accuracy epoch:%d] val_acc:%f'
              % (epoch + 1,  val_top1.avg))
        return val_top1.avg


if __name__ == '__main__':
    main()
