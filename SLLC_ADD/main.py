import time
import shutil
import argparse
import csv
import torch
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.enabled = False

from tqdm import tqdm
from autolab_core import YamlConfig

from utils import *
import smp_1


def main(dataloaders, datloaders_smp, audio_net, base_logger, writer, config, args, model_type, ckpt_path):
    model_parameters = [*audio_net.parameters()]
    optimizer, scheduler = get_optimizer_scheduler(model_parameters, config['OPTIMIZER'], config['SCHEDULER'])

    test_best_score = 0
    test_epoch_best = 0

    binary = np.load('D:/daicwoz/audiofile_daic/train/phq_binary_gt.npy', allow_pickle=True)
    binary = binary.squeeze(-1)
    softmax = torch.nn.Softmax(dim=1)
    alpha = 0.3
    beta = 1 - alpha
    mil_count = 1


    for epoch in range(config['EPOCHS']):
        log_and_print(base_logger, f'Epoch: {epoch}  Current Best Test: {test_best_score} at epoch {test_epoch_best}')

        for mode in ['train', 'test']:
            mode_start_time = time.time()

            phq_binary_gt = []
            phq_binary_pred = []
  
            if mode == 'train':
                if epoch>=config['UPDATE_EPOCHS']:
                    feature = np.zeros([7742, 512])
                    pred = np.zeros([7742, 2])
                    delta = 1.2 + 0.15 * max(epoch - config['UPDATE_EPOCHS'], 0)
                    global prototypeslist

                    audio_net.eval()
                    torch.set_grad_enabled(False)
                    with torch.no_grad():
                        if epoch==5:
                            feature_0 = []
                            feature_1 = []
                            for data_lc_0 in tqdm(dataloaders_smp['nd']):
                                _, x_0 = audio_net(data_lc_0['audio'].to(args.device))
                                x_0 = x_0.cpu().numpy()
                                feature_0.append(x_0)
                            for data_lc_1 in tqdm(dataloaders_smp['d']):
                                _, x_1 = audio_net(data_lc_1['audio'].to(args.device))
                                x_1 = x_1.cpu().numpy()
                                feature_1.append(x_1)
                            feature_0 = np.concatenate(feature_0, axis=0)
                            feature_1 = np.concatenate(feature_1, axis=0)
                            prototypeslist = smp_1.get_prototypes(feature_0, feature_1, k=2, p=6)

                        for data_t in tqdm(dataloaders[mode]):
                            index = data_t['num'].squeeze(-1)
                            probs, x = audio_net(data_t['audio'].to(args.device))
                            x = x.cpu().numpy()
                            probs = probs.cpu().numpy()
                            feature[index, :] = x
                            pred[index, :] = probs

                        y_pseudo_all = smp_1.produce_pseudo_labels(prototypeslist, feature)
                        y_lrt_all = lrt_flip_scheme(pred, binary, delta)

                        y_corrected = alpha * y_pseudo_all + beta * y_lrt_all
                        y_file = "D:/daicwoz/audiofile_daic/train/softlabel.npy"
                        np.save(y_file, y_corrected)

            if mode == 'train':
                audio_net.train()
                torch.set_grad_enabled(True)
            else:
                audio_net.eval()
                torch.set_grad_enabled(False)

            total_loss = 0
            log_interval_loss = 0
            log_interval = 80
            batch_number = 0
            n_batches = len(dataloaders[mode])
            batches_start_time = time.time()

            y_file = "D:/daicwoz/audiofile_daic/train/softlabel.npy"
            y_tilde = np.load(y_file, allow_pickle=True)

            for i, data in enumerate(tqdm(dataloaders[mode])):
                batch_size = data['ID'].size(0)
                index = data['num'].squeeze(-1)
                phq_binary_gt.extend(data['phq_binary_gt'].numpy().astype(float))  # 1D list

                if mode == 'train':

                    # choose the right GT for criterion based on prediciton type
                    gt = get_gt(data, config['EVALUATOR']['PREDICT_TYPE']).squeeze(-1)

                    criterion = get_criterion(config['CRITERION'], args)

                    audio = data['audio']
                    probs, x = audio_net(audio.to(args.device))
                    if epoch >= config['UPDATE_EPOCHS']:
                        last_y = y_tilde[index, :]
                        last_y = torch.FloatTensor(last_y)
                        last_y = softmax(last_y.to(args.device))
                        lc = torch.mean(last_y * (torch.log(last_y) - torch.log(probs)))
                        le = -torch.mean(torch.mul(probs, torch.log(probs)))
                        loss = lc + 0.1 * le
                    else:
                        loss = compute_loss(criterion, probs, gt, config['EVALUATOR'], args,
                                            use_soft_label=config['CRITERION']['USE_SOFT_LABEL'])
                    optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(models_parameters, max_norm=2.0, norm_type=2)
                    optimizer.step()

                else:
                    # for test set, only do prediction
                    audio = data['audio']
                    probs, x = audio_net(audio.to(args.device))

                # predict the final score

                pred_score = compute_score(probs, args)

                phq_binary_pred.extend([pred_score[i].item() for i in range(batch_size)])

                if mode == 'train':
                    # information per batch
                    total_loss += loss.item()
                    log_interval_loss += loss.item()
                    if batch_number % log_interval == 0 and batch_number > 0:
                        lr = scheduler.get_last_lr()[0]
                        ms_per_batch = (time.time() - batches_start_time) * 1000 / log_interval
                        current_loss = log_interval_loss / log_interval
                        print(f'| epoch {epoch:3d} | {mode} | {batch_number:3d}/{n_batches:3d} batches | '
                              f'LR {lr:7.6f} | ms/batch {ms_per_batch:5.2f} | loss {current_loss:8.5f} |')

                        # tensorboard
                        writer.add_scalar('Loss_per_{}_batches/{}'.format(log_interval, mode),
                                          current_loss, epoch * n_batches + batch_number)

                        log_interval_loss = 0
                        batches_start_time = time.time()
                else:
                    # for test set we don't need to calculate the loss so just leave it 'nan'
                    total_loss = np.nan

                batch_number += 1

            # print('PHQ Binary prediction: {}'.format(phq_binary_pred[:20]))

            # print('PHQ Binary ground truth: {}'.format(phq_binary_gt[:20]))

            average_loss = total_loss / n_batches
            lr = scheduler.get_last_lr()[0]
            s_per_mode = time.time() - mode_start_time
            accuracy, correct_number = get_accuracy(phq_binary_gt, phq_binary_pred)

            # store information in logger and print
            print('-' * 110)
            msg = ('  End of {0}:\n  | time: {1:8.3f}s | LR: {2:7.6f} | Average Loss: {3:8.5f} | Accuracy: {4:5.2f}%'
                   ' ({5}/{6}) |').format(mode, s_per_mode, lr, average_loss, accuracy * 100, correct_number,
                                          len(phq_binary_gt))
            log_and_print(base_logger, msg)
            print('-' * 110)

            # tensorboard
            writer.add_scalar('Loss_per_epoch/{}'.format(mode), average_loss, epoch)
            writer.add_scalar('Accuracy/{}'.format(mode), accuracy * 100, epoch)
            writer.add_scalar('Learning_rate/{}'.format(mode), lr, epoch)

            # Calculating additional evaluation scores
            log_and_print(base_logger, '  Output Scores:')

            # confusion matrix
            [[tn, fp], [fn, tp]] = standard_confusion_matrix(phq_binary_gt, phq_binary_pred)

            msg = (f'  - Confusion Matrix:\n'
                   '    -----------------------\n'
                   f'    | TN: {tn:4.0f} | FP: {fp:4.0f} |\n'
                   '    -----------------------\n'
                   f'    | FN: {fn:4.0f} | TP: {tp:4.0f} |\n'
                   '    -----------------------')
            log_and_print(base_logger, msg)

            # classification related
            precision, undepressed_recall, depressed_recall, f1_score, UAR = get_classification_scores(phq_binary_gt,
                                                                                                       phq_binary_pred)
            msg = ('  - Classification:\n'
                   '      precision: {0:6.4f}\n'
                   '      undepressed_recall: {1:6.4f}\n'
                   '      depressed_recall: {2:6.4f}\n'
                   '      f1_score: {3:6.4f}\n'
                   '      UAR: {4:6.4f}').format(precision, undepressed_recall, depressed_recall, f1_score, UAR)
            log_and_print(base_logger, msg)

            '''
            # regression related
            mae, mse, rmse, r2 = get_regression_scores(phq_score_gt, phq_score_pred)
            msg = ('  - Regression:\n'
                   '      MAE: {0:7.4f}\n'
                   '      MSE: {1:7.4f}\n'
                   '      RMSE: {2:7.4f}\n'
                   '      R2: {3:7.4f}\n').format(mae, mse, rmse, r2)
            log_and_print(base_logger, msg)
            '''
            # Calculate a Spearman correlation coefficien
            rho, p = stats.spearmanr(phq_binary_gt, phq_binary_pred)  # phq_binary_gt, phq_binary_pred
            msg = ('  - Correlation:\n'
                   '      Spearman correlation: {0:8.6f}\n').format(rho)
            log_and_print(base_logger, msg)

            # store the model score
            if mode == 'train':
                train_model_score = UAR
            elif mode == 'test':
                test_model_score = UAR

            # tensorboard
            writer.add_scalars(f'Classification/{mode}/Scores', {'Precision': precision,
                                                                 'UAR': UAR,
                                                                 'F1_score': f1_score}, epoch)
            '''
            writer.add_scalars('Regression/Scores', {'MAE': mae,
                                                     'MSE': mse,
                                                     'RMSE': rmse,
                                                     'R2': r2}, epoch)
            '''
            writer.add_scalar('Spearman_correlation/{}'.format(mode), rho, epoch)
        if test_model_score > test_best_score:
            test_best_score = test_model_score
            test_epoch_best = epoch

            msg = (f'--------- New best found at epoch {epoch} !!! ---------\n'
                   f'- train score: {train_model_score:8.6f}\n'
                   f'- test score: {test_model_score:8.6f}\n'
                   f'--------- New best found at epoch {epoch} !!! ---------\n')
            log_and_print(base_logger, msg)

            if epoch == 0:
                with open(os.path.join(ckpt_path, 'pred_binary_0.0005_0.3.csv'), 'w', newline='') as csvfile:
                    wwww = csv.writer(csvfile)
                    wwww.writerow(phq_binary_pred)
            else:
                with open(os.path.join(ckpt_path, 'pred_binary_0.0005_0.3.csv'), 'a+', newline='') as csvfile:
                    wwwww = csv.writer(csvfile)
                    wwwww.writerow(phq_binary_pred)

            if args.save:
                timestamp = datetime.utcnow().strftime('%Y-%m-%d_%H%M%S')
                file_path = os.path.join(ckpt_path,
                                         '{}_{}_{}_{:6.4f}_0.3.pt'.format(model_type, config['OPTIMIZER']['LR'],epoch,
                                                                                test_model_score))

                torch.save({'epoch': epoch,
                            'audio_net': audio_net.state_dict(),
                            'best_score': test_model_score},
                           file_path)

        # update lr with scheduler
        scheduler.step()
