import argparse
import os
import time
import random
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
import network
import torch.optim as optim
import pre_process as prep
from data_list import ImageList
import lr_schedule
from logger import Logger
import numpy as np
from tensorboardX import SummaryWriter
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


def val(data_set_loader, base_net, f1_net, f2_net, test_10crop, config, num_iter):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(data_set_loader['test'][i]) for i in range(10)]
            for i in range(len(data_set_loader['test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()

                outputs = []
                for j in range(10):
                    feature = base_net(inputs[j])
                    predict_out1 = f1_net(x=feature, alpha=0)
                    predict_out2 = f2_net(x=feature, alpha=0)
                    predict_out = predict_out1 + predict_out2
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)

                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), dim=0)
                    all_label = torch.cat((all_label, labels.float()), dim=0)
        else:
            iter_test = iter(data_set_loader['test'])
            for i in range(len(data_set_loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()

                feature = base_net(inputs)
                predict_out1 = f1_net(feature, 0)
                predict_out2 = f2_net(feature, 0)
                outputs = predict_out1 + predict_out2

                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if config['is_writer']:
        config['writer'].add_scalars('test', {'test error': 1.0 - accuracy,
                                              'acc': accuracy * 100.0},
                                     num_iter)
    return accuracy * 100.0


def train(config):
    # set pre-process
    prep_dict = {'source': prep.image_train(**config['prep']['params']),
                 'target': prep.image_train(**config['prep']['params'])}
    if config['prep']['test_10crop']:
        prep_dict['test'] = prep.image_test_10crop(**config['prep']['params'])
    else:
        prep_dict['test'] = prep.image_test(**config['prep']['params'])

    # prepare data
    data_set = {}
    data_set_loader = {}
    data_config = config['data']
    data_set['source'] = ImageList(open(data_config['source']['list_path']).readlines(),
                                   transform=prep_dict['source'])
    data_set_loader['source'] = torch.utils.data.DataLoader(data_set['source'],
                                                            batch_size=data_config['source']['batch_size'],
                                                            shuffle=True, num_workers=4, drop_last=True)
    data_set['target'] = ImageList(open(data_config['target']['list_path']).readlines(),
                                   transform=prep_dict['target'])
    data_set_loader['target'] = torch.utils.data.DataLoader(data_set['target'],
                                                            batch_size=data_config['target']['batch_size'],
                                                            shuffle=True, num_workers=4, drop_last=True)
    if config['prep']['test_10crop']:
        data_set['test'] = [ImageList(open(data_config['test']['list_path']).readlines(),
                                      transform=prep_dict['test'][i])
                            for i in range(10)]
        data_set_loader['test'] = [torch.utils.data.DataLoader(dset, batch_size=data_config['test']['batch_size'],
                                                               shuffle=False, num_workers=4)
                                   for dset in data_set['test']]
    else:
        data_set['test'] = ImageList(open(data_config['test']['list_path']).readlines(),
                                     transform=prep_dict['test'])
        data_set_loader['test'] = torch.utils.data.DataLoader(data_set['test'],
                                                              batch_size=data_config['test']['batch_size'],
                                                              shuffle=False, num_workers=4)

    # set base network
    net_config = config['network']
    base_net = net_config['name']()  # res50
    base_net = base_net.cuda()
    # add domain, classifier network
    class_num = config['network']['params']['class_num']
    ad_net = network.Domain(in_features=base_net.output_num(), hidden_size=1024)
    f1_net = network.Classifier(in_features=base_net.output_num(), hidden_size=128, class_num=class_num)
    f2_net = network.Classifier(in_features=base_net.output_num(), hidden_size=128, class_num=class_num)
    ad_net = ad_net.cuda()
    f1_net = f1_net.cuda()
    f2_net = f2_net.cuda()
    parameter_list = base_net.get_parameters() + ad_net.get_parameters() \
                     + f1_net.get_parameters() + f2_net.get_parameters()

    # multi gpus
    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        base_net = nn.DataParallel(base_net, device_ids=[int(i) for i in gpus])
        f1_net = nn.DataParallel(f1_net, device_ids=[int(i) for i in gpus])
        f2_net = nn.DataParallel(f2_net, device_ids=[int(i) for i in gpus])
        ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in gpus])

    # set optimizer
    optimizer_config = config['optimizer']
    optimizer = optimizer_config['type'](parameter_list, **(optimizer_config['optim_params']))
    schedule_param = optimizer_config['lr_param']
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config['lr_type']]

    # set loss
    class_criterion = nn.CrossEntropyLoss()
    transfer_criterion = nn.BCELoss()
    loss_params = config['loss']

    # train
    len_train_source = len(data_set_loader['source'])
    len_train_taget = len(data_set_loader['target'])
    best_acc = 0.0
    since = time.time()
    for num_iter in tqdm(range(config['max_iter'])):
        if num_iter % config['val_iter'] == 0:
            base_net.train(False)
            f1_net.train(False)
            f2_net.train(False)
            temp_acc = val(data_set_loader, base_net, f1_net, f2_net,
                           test_10crop=config['prep']['test_10crop'],
                           config=config, num_iter=num_iter)
            if temp_acc > best_acc:
                best_acc = temp_acc
            log_str = 'iter: {:d}, accu: {:.4f}\ntime: {:.4f}'.format(num_iter, temp_acc, time.time() - since)
            config['logger'].logger.debug(log_str)
            config['results'][num_iter].append(temp_acc)
        if num_iter % config['snapshot_iter'] == 0:
            # TODO
            pass

        base_net.train(True)
        f1_net.train(True)
        f2_net.train(True)
        ad_net.train(True)

        optimizer = lr_scheduler(optimizer, num_iter, **schedule_param)
        optimizer.zero_grad()

        if num_iter % len_train_source == 0:
            iter_source = iter(data_set_loader['source'])
        if num_iter % len_train_taget == 0:
            iter_target = iter(data_set_loader['target'])
        input_source, label_source = iter_source.next()
        input_target, _ = iter_target.next()
        input_source, label_source, input_target = input_source.cuda(), label_source.cuda(), input_target.cuda()
        inputs = torch.cat((input_source, input_target), 0)
        batch_size = len(label_source)

        # class-wise adaptation
        features = base_net(inputs)
        alpha = 2. / (1. + np.exp(-10 * float(num_iter / config['max_iter']))) - 1
        output1 = f1_net(features, alpha)
        output2 = f2_net(features, alpha)
        output_s1 = output1[:batch_size, :]
        output_s2 = output2[:batch_size, :]
        output_t1 = output1[batch_size:, :]
        output_t2 = output2[batch_size:, :]
        output_t1 = F.softmax(output_t1)
        output_t2 = F.softmax(output_t2)

        inconsistency_loss = torch.mean(torch.abs(output_t1 - output_t2))
        classifier_loss1 = class_criterion(output_s1, label_source)
        classifier_loss2 = class_criterion(output_s2, label_source)
        classifier_loss = classifier_loss1 + classifier_loss2

        # domain-wise adaptation
        domain_labels = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
        domain_output = ad_net(features, alpha)
        transfer_loss = transfer_criterion(domain_output, domain_labels)

        total_loss = classifier_loss + loss_params['domain_off'] * transfer_loss \
                     - loss_params['dis_off'] * inconsistency_loss
        total_loss.backward()
        optimizer.step()

        if num_iter % config['val_iter'] == 0:
            print('class:', classifier_loss.item(),
                  'domain:', transfer_loss.item() * loss_params['domain_off'],
                  'inconsistency:', inconsistency_loss.item() * loss_params['dis_off'],
                  'total:', total_loss.item())
            if config['is_writer']:
                config['writer'].add_scalars('train', {'total': total_loss.item(), 'class': classifier_loss.item(),
                                                       'domain': transfer_loss.item() * loss_params['domain_off'],
                                                       'inconsistency': inconsistency_loss.item() * loss_params[
                                                           'dis_off']},
                                             num_iter)
    if config['is_writer']:
        config['writer'].close()

    return best_acc


def empty_dict(config):
    config['results'] = {}
    for i in range(config['max_iter'] // config['val_iter'] + 1):
        key = config['val_iter'] * i
        config['results'][key] = []
    config['results']['best'] = []


def print_dict(config):
    for i in range(config['max_iter'] // config['val_iter'] + 1):
        key = config['val_iter'] * i
        log_str = 'iter: {:d}, average: {:.4f}'.format(key, np.average(config['results'][key]))
        config['logger'].logger.debug(log_str)
    log_str = 'best, average: {:.4f}'.format(np.average(config['results']['best']))
    config['logger'].logger.debug(log_str)
    config['logger'].logger.debug('-' * 100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint Adversarial Domain Adaptation')
    parser.add_argument('--seed', type=int, default=1, help='manual seed')
    parser.add_argument('--gpu', type=str, nargs='?', default='0', help='device id to run')
    parser.add_argument('--resnet', type=str, default='ResNet50', help='Options: 50')
    parser.add_argument('--data_set', type=str, default='office', help='Options: office,clef')
    parser.add_argument('--source_path', type=str, default='../data/office/amazon_list.txt', help='The source list')
    parser.add_argument('--target_path', type=str, default='../data/office/webcam_list.txt', help='The target list')
    parser.add_argument('--max_iter', type=int, default=20001, help='max iterations')
    parser.add_argument('--val_iter', type=int, default=1000, help='interval of two continuous test phase')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=36, help='mini batch size')
    parser.add_argument('--output_path', type=str, default='checkpoint/', help='save log/scalar/model file path')
    parser.add_argument('--log_file', type=str, default='log', help='log file name')
    parser.add_argument('--is_writer', type=bool, default=True, help='whether record for tensorboard')
    parser.add_argument('--snapshot_iter', type=int, default=5000, help='interval of two continuous output model')
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = {'seed': args.seed, 'gpu': args.gpu, 'max_iter': args.max_iter, 'val_iter': args.val_iter,
              'is_writer': args.is_writer, 'snapshot_iter': args.snapshot_iter,
              'output_path': args.output_path,
              'loss': {'domain_off': 1.0, 'dis_off': 1.0},
              'prep': {'test_10crop': True, 'params': {'resize_size': 256, 'crop_size': 224}},
              'network': {'name': network.ResBase, 'params': {'resnet_name': args.resnet, 'class_num': 31}},
              'optimizer': {'type': optim.SGD,
                            'optim_params': {'lr': args.lr, 'momentum': 0.9, 'weight_decay': 0.0005, 'nesterov': True},
                            'lr_type': 'inv',
                            'lr_param': {'lr': args.lr, 'gamma': 0.0003, 'power': 0.75}}, 'data_set': args.data_set,
              'data': {
                  'source': {'name': args.source_path.split('/')[-1].split('_')[0], 'list_path': args.source_path,
                             'batch_size': args.batch_size},
                  'target': {'name': args.target_path.split('/')[-1].split('_')[0], 'list_path': args.target_path,
                             'batch_size': args.batch_size},
                  'test': {'list_path': args.target_path, 'batch_size': args.batch_size}}}

    if not os.path.exists(config['output_path']):
        os.makedirs(config['output_path'])
    if config['is_writer']:
        config['writer'] = SummaryWriter(log_dir=config['output_path'] + '/scalar/', )
    config['logger'] = Logger(logroot=config['output_path'], filename=args.log_file, level='debug')
    config['logger'].logger.debug(str(config))

    empty_dict(config)
    config['results']['best'].append(train(config))
    print_dict(config)
