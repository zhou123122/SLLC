from itertools import count
import os
import sys
import random
import logging
import numpy as np
from datetime import datetime
from scipy import stats
from sklearn import metrics
import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler, WeightedRandomSampler, DataLoader
from torchvision import transforms
# local functions
from dataset.dataset import DepressionDataset, ToTensor, SMPDataset
from models.CONVLSTM.convlstm import ConvLSTM_Audio

def init_seed(manual_seed):
    """
    Set random seed for torch and numpy.
    """
    random.seed(manual_seed)
    os.environ['PYTHONHASHSEED'] = str(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    tf.random.set_seed(manual_seed)

    torch.cuda.manual_seed_all(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(filepath, log_title):
    logger = logging.getLogger(filepath)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info('-' * 54 + log_title + '-' * 54)
    return logger


def log_and_print(logger, msg):
    logger.info(msg)
    print(msg)


def worker_init_fn(worker_id):
    """
    Init worker in dataloader.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_sampler_phq_binary(phq_binary_gt):
    # sampler for phq_binary_gt
    class_sample_count = np.unique(phq_binary_gt, return_counts=True)[1]
    weight = 1. / class_sample_count
    samples_weight = weight[phq_binary_gt]
    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


def get_sampler_phq_score(phq_score_gt):
    class_sample_ID, class_sample_count = np.unique(phq_score_gt, return_counts=True)
    weight = 1. / class_sample_count
    samples_weight = np.zeros(phq_score_gt.shape)
    for i, sample_id in enumerate(class_sample_ID):
        indices = np.where(phq_score_gt == sample_id)[0]
        value = weight[i]
        samples_weight[indices] = value
    samples_weight = torch.from_numpy(samples_weight).double().squeeze(-1)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


def get_dataloaders(data_config):
    dataloaders = {}
    for mode in ['train', 'test']:
        if mode == 'train':
            trainset = DepressionDataset(data_config[f'{mode}_ROOT_DIR'.upper()], mode,
                                         transform=transforms.Compose([ToTensor(mode)]))
            sampler = get_sampler_phq_score(trainset.phq_binary_gt)
            # sampler = RandomSampler(trainset)
            dataloaders[mode] = DataLoader(trainset,
                                           batch_size=data_config['BATCH_SIZE'],
                                           num_workers=data_config['NUM_WORKERS'],
                                           sampler=sampler
                                           )
        else:
            # for test dataset, we don't need shuffle, sampler and augmentation
            dataset = DepressionDataset(data_config[f'{mode}_ROOT_DIR'.upper()], mode,
                                        transform=transforms.Compose([ToTensor(mode)]))
            dataloaders[mode] = DataLoader(dataset,
                                           batch_size=data_config['BATCH_SIZE'],
                                           num_workers=data_config['NUM_WORKERS'])
    return dataloaders

def get_dataloaders_smp(data_config):
    dataloaders_smp = {}
    for mode in ['nd','d']:
        dataset = SMPDataset(data_config[f'SMP_ROOT_DIR'.upper()], mode, transform=transforms.Compose([ToTensor('smp')]))
        dataloaders_smp[mode] = DataLoader(dataset, batch_size=32, num_workers=data_config['NUM_WORKERS'])
    return dataloaders_smp

def get_models(model_config, args):
    audio_net = ConvLSTM_Audio(input_dim=model_config['AUDIO_NET']['INPUT_DIM'],
                               output_dim=model_config['AUDIO_NET']['OUTPUT_DIM'],
                               conv_hidden=model_config['AUDIO_NET']['CONV_HIDDEN'],
                               lstm_hidden=model_config['AUDIO_NET']['LSTM_HIDDEN'],
                               num_layers=model_config['AUDIO_NET']['NUM_LAYERS'],
                               output=model_config['EVALUATOR']['CLASSES_RESOLUTION'],
                               activation=model_config['AUDIO_NET']['ACTIVATION'],
                               norm=model_config['AUDIO_NET']['NORM'],
                               dropout=model_config['AUDIO_NET']['DROPOUT'])

    if len(args.gpu.split(',')) > 1:
        audio_net = nn.DataParallel(audio_net)

    # move to GPU
    audio_net = audio_net.to(args.device)

    return audio_net


def get_criterion(config, args):
    criterion = nn.CrossEntropyLoss()

    return criterion


def get_optimizer_scheduler(model_parameters, optimizer_config, scheduler_config, use_sgd=False):
    # get optimizer and scheduler

    optimizer = torch.optim.Adam(model_parameters, betas=(0.9, 0.999),eps=1e-06,
                                 lr=optimizer_config['LR'],
                                 weight_decay=optimizer_config['WEIGHT_DECAY'])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=scheduler_config['STEP_SIZE'],
                                                gamma=scheduler_config['GAMMA'])
    return optimizer, scheduler


def get_gt(data, predict_type):
    if predict_type == 'phq-subscores':
        gt = data['phq_subscores_gt']

    elif predict_type == 'phq-score':
        gt = data['phq_score_gt']

    elif predict_type == 'phq-binary':
        gt = data['phq_binary_gt']

    else:
        raise AssertionError("Unknown 'PREDICT_TYPE' for evaluator!", predict_type)

    return gt


def compute_score(probs, args):
    # factor = evaluator_config['N_CLASSES'] / evaluator_config['CLASSES_RESOLUTION']
    score_pred = (probs.argmax(dim=1)).to(int).to(float)

    return score_pred.to(args.device)


def convert_soft_gt(gt, evaluator_config):
    if evaluator_config['PREDICT_TYPE'] == 'phq-subscores':
        # gt: 1D array with shape 1 x num_subscores
        # Each subscore choose a score from [0, 1, 2, 3], we normalize it into 0 ~ config['CLASSES_RESOLUTION']
        factor = (evaluator_config['N_CLASSES'] - 1) / (evaluator_config['CLASSES_RESOLUTION'] - 1)
        tmp = [stats.norm.pdf(np.arange(evaluator_config['CLASSES_RESOLUTION']), loc=score / factor,
                              scale=evaluator_config['STD']).astype(np.float32) for score in gt]

        tmp = np.stack(tmp)  # shape: (num_subscores, class_resolution)

    else:
        # gt: a float value
        tmp = stats.norm.pdf(np.arange(evaluator_config['CLASSES_RESOLUTION']), loc=gt,
                             scale=evaluator_config['STD']).astype(np.float32)  # shape: (class_resolution, )

    return torch.from_numpy(tmp / tmp.sum(axis=-1, keepdims=True))


def get_soft_gt(gt, evaluator_config):
    soft_gt = torch.tensor([[]])

    # iterate through each batch
    for i in range(len(gt)):

        current_gt = gt[i]
        converted_current_gt = convert_soft_gt(current_gt, evaluator_config)
        if i == 0:
            soft_gt = converted_current_gt.unsqueeze(dim=0)
        else:
            soft_gt = torch.cat([soft_gt, converted_current_gt.unsqueeze(dim=0)], dim=0)

    return soft_gt  # shape (batch, class_resolution) or (batch, num_subscores, class_resolution)


def compute_loss(criterion, probs, gt, evaluator_config, args, use_soft_label=False):
    if use_soft_label:  # in this case, criterion should be nn.KLDivLoss()

        # convert GT to soft label
        soft_gt = get_soft_gt(gt, evaluator_config)

        if evaluator_config['PREDICT_TYPE'] == 'phq-subscores':
            loss = sum([criterion(torch.log(probs[i]), soft_gt[:, i].to(args.device))
                        for i in range(evaluator_config['N_SUBSCORES'])])
        else:
            loss = criterion(torch.log(probs), soft_gt.to(args.device))

    else:  # in this case, criterion should be  nn.CrossEntropyLoss() with weights

        if evaluator_config['PREDICT_TYPE'] == 'phq-subscores':
            # convert to shape (batch size,  number of subscores, class resolution)
            pred_prob = torch.stack([prob for prob in probs], dim=1)
            # compute loss: make sure dem-1 of prediction is softmax probabilities
            loss = criterion(pred_prob.permute(0, 2, 1).contiguous(),
                             gt.type(torch.LongTensor).to(args.device))
        else:
            loss = criterion(probs, gt.type(torch.LongTensor).to(args.device))

    return loss


def standard_confusion_matrix(gt, pred):
    [[tn, fp], [fn, tp]] = metrics.confusion_matrix(np.asarray(gt), np.asarray(pred))
    return np.array([[tn, fp], [fn, tp]])


def get_accuracy(gt, pred):
    [[tn, fp], [fn, tp]] = standard_confusion_matrix(gt, pred)
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    correct_number = tp + tn
    return accuracy, correct_number


def get_classification_scores(gt, pred):
    [[tn, fp], [fn, tp]] = standard_confusion_matrix(gt, pred)
    precision = tp / (tp + fp)
    depressed_recall = tp / (tp + fn)
    undepressed_recall = tn / (tn + fp)
    f1_score = 2 * (precision * depressed_recall) / (precision + depressed_recall)
    UAR = (undepressed_recall + depressed_recall) / 2
    return precision, undepressed_recall, depressed_recall, f1_score, UAR


def lrt_flip_scheme(pred_softlabels_bar, y_tilde, delta):

    ntrain = pred_softlabels_bar.shape[0]
    num_class = pred_softlabels_bar.shape[1]
    clean_softlabels = np.zeros([ntrain, num_class])

    for i in range(ntrain):
        cond_1 = (not pred_softlabels_bar[i].argmax()==y_tilde[i])
        cond_2 = (pred_softlabels_bar[i].max()/pred_softlabels_bar[i][int(y_tilde[i])] > delta)
        if cond_1 and cond_2:
            y_tilde[i] = pred_softlabels_bar[i].argmax()

    clean_softlabels[:, 0] = y_tilde
    clean_softlabels[:, 1] = 1-clean_softlabels[:, 0]

    return clean_softlabels




