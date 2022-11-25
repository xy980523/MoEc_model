import sys
import os
import json
import argparse
import csv
import math
import numpy as np
from tabulate import tabulate

parser = argparse.ArgumentParser()

# python get_log.py --prefix /mnt/unilm/moe/finetune --ckp-prefix /mnt/unilm/moe/finetune \
#--folders pt_moec64_c0d0/checkpoint_1_120000-ft \
#--mean --step 120 --output /home/shaohanh/moe_nv/scripts/t.txt


# common args
parser.add_argument('--prefix', type=str)
parser.add_argument('--ckp-prefix', default="", type=str)
parser.add_argument('--folders', type=str)
parser.add_argument('--mean', action='store_true', default=False)
parser.add_argument('--output', type=str)
parser.add_argument('--step', type=int, default=125)
parser.add_argument('--model', type=str, default='base')

args = parser.parse_args()

ckps = [f'checkpoint_1_{args.step}000.pt']  # 30 epoch for 128k-vocab data

glue_lr1 = ['1e-5', '2e-5', '3e-5', '4e-5']
glue_lr2 = ['1e-5','2e-5', '3e-5', '4e-5', '5e-5']
squad_lr = ['2e-5', '3e-5', '4e-5']
if args.model == 'large':
    glue_lr1 = ['5e-6', '7e-6', '1e-5', '2e-5']
    glue_lr2 = ['7e-6', '1e-5',  '2e-5', '3e-5']
    squad_lr = ['7e-6', '1e-5',  '2e-5', '3e-5']
elif args.model == 'xlarge':
    glue_lr1 = ['3e-6', '5e-6', '7e-6', '1e-5']
    glue_lr2 = ['5e-6', '7e-6',  '1e-5', '2e-5']
    squad_lr = ['5e-6', '7e-6',  '1e-5', '2e-5']


def mean(res):
    return sum(res) / len(res)


def median(res):
    return np.median(res)


agg_fun = mean if args.mean else median
folders = args.folders.split(',')


def get_glue(f):
    def get_from_line(line, metric):
        ss = line.split('|')
        for s in ss:
            ts = s.strip().split(' ')
            if ts[0] == metric:
                return float(ts[1])
        return -1

    def get_from_file(filename, keyword, metric):
        try:
            if not os.path.isfile(filename):
                # print('{} not found'.format(filename))
                return -1
            with open(filename, "r") as input:
                ret = -1
                check_cnt = 0
                flag = False
                for line in input.readlines():
                    if "Now we only have" in line and not flag:
                            # import pudb;pu.db;
                            flag = True
                            # print("OK")
                    if keyword in line and 'valid on' in line:
                        cur_res = get_from_line(line, metric)
                        
                        # ret = max(cur_res, ret)
                        # use to get the last result
                        if flag:
                            ret = max(cur_res, ret)
                        else:
                            ret = cur_res
                    # if 'Loaded checkpoint' in line:
                    #     check_cnt += 1
                    if 'done training' in line:
                        check_cnt += 1
                check_cnt = 1
                if check_cnt == 1:
                    if ret <= 0 or ret != ret:
                        print("{}: accuracy is {}".format(filename, ret))
                        ret = 0
                    return ret
                else:
                    return -1
        except Exception as e:
            return -1

    def one_point(prefix, ckp_prefix, folder, name, ckp, keyword, metric, agg_fun):
        if name in ['MNLI', 'QNLI', 'QQP', 'SST-2']:
            epochs = ['3','5']
            warmups = ['16']
            # warmups = ['10']
            bszs = ['32']
            lrs = ['2e-5']
        else:
            epochs = ['3','5','10']
            warmups = ['16']
            bszs = ['32']
            lrs = ['2e-5']
        # seeds = ['1', '2', '3', '4', '5']
        seeds = ['1', '2']
        # seeds = ['1']
        best_res = 0
        best_setting = None
        valid = True
        ckp_filename = ckp_prefix + '/' + folder + '/' + ckp
        #if ckp_prefix and not os.path.isfile(ckp_filename):
        #    print('Checkpoint {} does not exist'.format(ckp_filename))
        #    return float('NaN'), False
        for epoch in epochs:
            for warmup in warmups:
                for bsz in bszs:
                    for lr in lrs:
                        total_ret = []
                        for seed in seeds:
                            # filename = prefix + '/' + folder + '/' + ckp[:-3] + '-ft/' + name + '/' + \
                            #     '{0}-{1}-{2}-{3}-{4}/train_log.txt'.format(
                            #         epoch, warmup, bsz, lr, seed)
                            filename = prefix + '/' + folder + '/'  + name + '/' + \
                                '{0}-{1}-{2}-{3}-{4}/train_log.txt'.format(
                                    epoch, warmup, bsz, lr, seed)
                            print(filename)
                            res = get_from_file(filename, keyword, metric)
                            if res < 0:
                                valid = False
                            total_ret.append(res)
                        print(folder, name, ckp, epoch, warmup,
                              bsz, lr, sorted(total_ret))
                        cur_res = agg_fun(total_ret)
                        if cur_res > best_res:
                            best_res = cur_res
        return best_res, valid
    names =  ['MNLI-m', 'MNLI-mm', 'QNLI', 'QQP', 'SST-2', 'CoLA', 'MRPC', 'RTE', 'STS-B']
    tasks = ['MNLI', 'MNLI', 'QNLI', 'QQP', 'SST-2', 'CoLA', 'MRPC', 'RTE', 'STS-B']
    #names = ['CoLA','MNLI-m', 'MNLI-mm', 'SST-2','QQP', 'QNLI', 'MRPC', 'RTE']
    #tasks = ['CoLA','MNLI', 'MNLI','SST-2','QQP', 'QNLI', 'MRPC', 'RTE']
    # names = ['MNLI-m', 'MNLI-mm']
    # tasks = ['MNLI', 'MNLI']
    keywords = ["'valid'" for _ in range(len(tasks))]
    # keywords[1] = "'valid1'"
    metrics = ['accuracy' for _ in range(len(tasks))]
    # metrics[5] = 'mcc'
    # metrics[-1] = 'pearson'
    result_dict = {}
    is_valid = {}
    for i in range(len(names)):
        result_dict[names[i]] = {}
        is_valid[names[i]] = {}
        for folder in folders:
            result_dict[names[i]][folder] = {}
            is_valid[names[i]][folder] = {}
            for ckp in ckps:
                step = int(ckp.split('.')[0].split('_')[-1]) // 1000
                res, valid = one_point(
                    args.prefix, args.ckp_prefix, folder, tasks[i], ckp, keywords[i], metrics[i], agg_fun)
                result_dict[names[i]][folder][step] = res
                is_valid[names[i]][folder][step] = valid

    for name in names:
        f.write(name + '\n')
        header = ','.join(folders)
        f.write('step,' + header + '\n')
        for ckp in ckps:
            step = int(ckp.split('.')[0].split('_')[-1]) // 1000
            to_write = [str(step)]
            for exp in folders:
                res = result_dict[name][exp][step]
                if is_valid[name][exp][step]:
                    to_write.append(str(res))
                elif res > 0:
                    to_write.append(':'+str(res))
                else:
                    to_write.append('')
            f.write(','.join(to_write) + '\n')
        f.write('\n')

    f.write('GLUE average' + '\n')
    header = ','.join(folders)
    f.write('step,' + header + '\n')
    for ckp in ckps:
        step = int(ckp.split('.')[0].split('_')[-1]) // 1000
        to_write = [str(step)]
        for exp in folders:
            sum_res = 0
            for name in names:
                if result_dict[name][exp][step] >= 0:
                    sum_res += result_dict[name][exp][step]
                elif math.isnan(result_dict[name][exp][step]):
                    sum_res = float('NaN')
                    break
                else:
                    sum_res = -1
                    break
            sum_res /= len(names)
            if math.isnan(sum_res):
                to_write.append('NaN')
            else:
                to_write.append(str(sum_res))
        f.write(','.join(to_write) + '\n')
    return result_dict


def get_squad(f, name):
    assert name == 'squad1' or name == 'squad2'

    def get_from_line(line, keyword):
        tokens = line.split('),')
        for token in tokens:
            if keyword in token:
                ret = float(token.split(',')[-1])
                return ret

    def get_from_file(filename):
        try:
            if not os.path.isfile(filename):
                return -1, -1
            check_cnt = 0
            with open(filename, "r") as input:
                best_f1 = -1
                best_em = -1
                for line in input.readlines():
                    if 'OrderedDict' in line:
                        f1 = get_from_line(line, "'best_f1'")
                        em = get_from_line(line, "'best_exact'")
                        best_f1 = max(f1, best_f1)
                        best_em = max(em, best_em)
                    # if 'Loaded checkpoint' in line:
                    #     check_cnt += 1
                    if 'done training' in line:
                        check_cnt += 1
                if check_cnt == 1:
                    return best_f1, best_em
                else:
                    return -1, -1
        except Exception as e:
            return -1, -1

    def one_point(prefix, folder, ckp):
        ckp_filename = args.ckp_prefix + '/' + folder + '/' + ckp
        # if args.ckp_prefix and not os.path.isfile(ckp_filename):
        #     print('Checkpoint {} does not exist'.format(ckp_filename))
        #     return float('NaN'), float('NaN'), False
        epochs = ['3']
        warmups = ['10']
        bszs = ['32']
        lrs = squad_lr
        # seeds = ['1', '2', '3', '4', '5']
        seeds = ['1', '2']
        # $DIR/$N_EPOCH-$WARMUP_RATIO-$BSZ-$LR-$SEED
        best_f1 = -1
        best_em = -1
        valid = True
        for epoch in epochs:
            for warmup in warmups:
                for bsz in bszs:
                    for lr in lrs:
                        total_f1 = []
                        total_em = []
                        for seed in seeds:
                            filename = prefix + '/' + folder  + \
                                '/{5}/{0}-{1}-{2}-{3}-{4}/train_log.txt'.format(
                                    epoch, warmup, bsz, lr, seed, name)
                            f1, em = get_from_file(filename)
                            total_f1.append(f1)
                            total_em.append(em)
                            if f1 < 0 or em < 0:
                                valid = False
                        print(folder, name, ckp, epoch, warmup, bsz,
                              lr, sorted(total_f1), sorted(total_em))
                        cur_f1 = agg_fun(total_f1)
                        cur_em = agg_fun(total_em)
                        best_f1 = max(best_f1, cur_f1)
                        best_em = max(best_em, cur_em)
        return best_f1, best_em, valid
    result_dict = {}
    is_valid = {}
    for folder in folders:
        result_dict[folder] = {}
        is_valid[folder] = {}
        for ckp in ckps:
            step = int(ckp.split('.')[0].split('_')[-1]) // 1000
            f1, em, valid = one_point(args.prefix, folder, ckp)
            result_dict[folder][step] = {'f1': f1, 'em': em}
            is_valid[folder][step] = valid
    header = ','.join(folders)
    f.write('step,' + header + '\n')
    f.write('{} F1, \n'.format(name))
    for ckp in ckps:
        step = int(ckp.split('.')[0].split('_')[-1]) // 1000
        to_write = [str(step)]
        for exp in folders:
            res = result_dict[exp][step]['f1']
            if res > 0:
                if is_valid[exp][step]:
                    to_write.append(str(res))
                else:
                    to_write.append(':'+str(res))
            else:
                to_write.append('')
        f.write(','.join(to_write) + '\n')
    f.write('\n')
    f.write('{} EM, \n'.format(name))
    for ckp in ckps:
        step = int(ckp.split('.')[0].split('_')[-1]) // 1000
        to_write = [str(step)]
        for exp in folders:
            res = result_dict[exp][step]['em']
            if res > 0:
                if is_valid[exp][step]:
                    to_write.append(str(res))
                else:
                    to_write.append(':'+str(res))
            else:
                to_write.append('')
        f.write(','.join(to_write) + '\n')
    f.write('\n')
    return result_dict


def print_summary_table(r_glue, r_squad):
    # exp_set = set(r_glue['MNLI-m'].keys()) | set(r_squad.keys())
    tb_list = []
    for exp in folders:
        # line = [exp, r_glue['MNLI-m'][exp][args.step], r_glue['MNLI-mm'][exp][args.step], r_squad[exp][args.step]['f1'], r_squad[exp][args.step]['em']]
        line = [exp,r_glue['CoLA'][exp][args.step],  r_glue['MNLI-m'][exp][args.step], r_glue['STS-B'][exp][args.step],r_glue['SST-2'][exp][args.step], r_glue['QQP'][exp][args.step], r_glue['QNLI'][exp][args.step], r_glue['MRPC'][exp][args.step], r_glue['RTE'][exp][args.step],r_squad[exp][args.step]['f1'], r_squad[exp][args.step]['em']]
        tb_list.append(line)
    # print(tabulate(tb_list, headers=[
    #       'Run', 'MNLI-m', 'MNLI-mm', 'SQuAD2-F1', 'SQuAD2-EM']))
    print(tabulate(tb_list, headers=[
          'Run', 'CoLA','MNLI-m', 'STS-B','SST-2', 'QQP', 'QNLI', 'MRPC', 'RTE', 'SQuAD2-F1', 'SQuAD2-EM' ]))



with open(args.output, 'w') as f:
    # import pudb;pu.db;
    r_glue = get_glue(f)
    # get_squad(f, 'squad1')
    # import pudb;pu.db;
    r_squad = get_squad(f, 'squad2')
    print_summary_table(r_glue, r_squad)
