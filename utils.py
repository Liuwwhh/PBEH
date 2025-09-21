import os
import torch
import random
import logging
import numpy as np
from torch.nn import functional as F

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def logger(args):
    '''
    '\033[0;34m%s\033[0m': blue
    :return:
    '''
    logger = logging.getLogger('Prompt')
    logger.setLevel(logging.DEBUG)
    log_floder_path = f'{args.output_dir}logs/{args.runid}/'
    os.makedirs(log_floder_path, exist_ok=True)
    log_path = log_floder_path + '/' + '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.log'.format(
        args.dataset_name, 
        args.bit, 
        args.prompt_mode, 
        args.prompt_extend_per_sample_number, 
        args.extend_bit, 
        args.learning_rate, 
        args.hash_cos_sim_loss, 
        args.similartity_main_loss, 
        args.quantify_loss, 
        args.contras_loss, 
        args.disstill_loss, 
        args.prompt_hash_main_loss, 
        args.error_samples_ratio
        )
    txt_log = logging.FileHandler(log_path)

    txt_log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', '%m/%d %H:%M:%S')
    txt_log.setFormatter(formatter)
    logger.addHandler(txt_log)

    print(f'log will be stored to {txt_log}')

    return logger


def creat_result_dict(args):
    I2T = np.zeros((args.num_tasks, args.num_tasks+1))
    T2I = np.zeros((args.num_tasks, args.num_tasks+1))
    result_dict = {
        'I2T' : I2T,
        'T2I' : T2I,
    }
    return result_dict


def save_result_dict(args, result_dict, csv_folder):
    # Average the data
    for i in range(args.num_tasks):
        map_sum_I2T = 0
        map_sum_T2I = 0
        for j in range(i+1):
            map_sum_I2T += result_dict['I2T'][i, j]
            map_sum_T2I += result_dict['T2I'][i, j]
        result_dict['I2T'][i, args.num_tasks] = map_sum_I2T / (i+1)
        result_dict['T2I'][i, args.num_tasks] = map_sum_T2I / (i+1)

    # Specify the CSV filepath
    os.makedirs(csv_folder, exist_ok=True)
    csv_file = csv_folder + '/' + '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv'.format(
        args.dataset_name, 
        args.bit, 
        args.prompt_mode, 
        args.prompt_extend_per_sample_number, 
        args.extend_bit, 
        args.learning_rate, 
        args.hash_cos_sim_loss, 
        args.similartity_main_loss, 
        args.quantify_loss, 
        args.contras_loss, 
        args.disstill_loss, 
        args.prompt_hash_main_loss, 
        args.error_samples_ratio
        )

    # Write the numpy matrix from the dictionary to a CSV file
    with open(csv_file, 'w') as f:
        for key, matrix in result_dict.items():
            # Write the data for each matrix to a file
            f.write(key + '\n')
            np.savetxt(f, matrix, delimiter=',', fmt='%f')
    print(f'The result is already stored under {csv_file}')


def train_select_prompt(args, sample, expert_prompt_pool, task_index):
    prompt = torch.cat(expert_prompt_pool[task_index], dim=0)
    prompt = F.normalize(prompt)
    select_prompt_sim_matrix = sample @ prompt.t()
    max_indices = torch.argmax(select_prompt_sim_matrix, dim=1)
    select_prompt = [expert_prompt_pool[task_index][q] for q in max_indices]
    select_prompt = torch.cat(select_prompt, dim=0)
    return select_prompt


def valid_select_prompt(args, sample, expert_prompt_pool, task_index):
    prompt = F.normalize(torch.cat([torch.cat(expert_prompt_pool[i], dim=0) for i in range(task_index+1)])).cuda()
    select_prompt_sim_matrix = sample @ prompt.t()
    max_indices = torch.argmax(select_prompt_sim_matrix, dim=1)
    max_indices = max_indices.cuda()
    select_prompt = []
    for q in max_indices:
        if (q/args.num_prompt).floor().to(torch.int) == 0:
            select_prompt.append(expert_prompt_pool[0][q%args.num_prompt])
        elif (q/args.num_prompt).floor().to(torch.int) == 1:
            select_prompt.append(expert_prompt_pool[1][q%args.num_prompt])
        elif (q/args.num_prompt).floor().to(torch.int) == 2:
            select_prompt.append(expert_prompt_pool[2][q%args.num_prompt])
        elif (q/args.num_prompt).floor().to(torch.int) == 3:
            select_prompt.append(expert_prompt_pool[3][q%args.num_prompt])
        else:
            select_prompt.append(expert_prompt_pool[4][q%args.num_prompt])
    select_prompt = torch.cat(select_prompt, dim=0)
    return select_prompt


def calc_map_k(qu_B, re_B, qu_L, re_L, topk=None):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = qu_L.shape[0]
    map = 0
    if topk is None:
        topk = re_L.shape[0]
    for iter in range(num_query):
        q_L = qu_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)
        gnd = (q_L.mm(re_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        hamm = calc_hammingDist(qu_B[iter, :], re_B)
        _, ind = torch.sort(hamm, stable=True)   # 默认稳定排序
        ind.squeeze_()
        gnd = gnd[ind]
        tgnd = gnd[:topk]
        tsum = torch.sum(tgnd)
        if tsum == 0:
            continue
        count = torch.arange(1, int(tsum) + 1).type(torch.float32)
        tindex = torch.nonzero(tgnd).squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.cuda()
        map = map + torch.mean(count / tindex)
    map = map / num_query
    return map


def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH


def compute_mode_center(h_matrix):
    mode_center = torch.sign(torch.sum(h_matrix, dim=0))  # (hash_length,)
    mode_center_ste = h_matrix.mean(dim=0) + (mode_center - h_matrix.mean(dim=0)).detach()
    return mode_center_ste


@torch.no_grad()
def extract_prompt_hash_test(args, hashing_model, iteration, init_prompt_number):
    if args.prompt_mode == 'specific':
        prompt_feature_image = hashing_model.image_net.attention(
            hashing_model.prompt_list[iteration][:init_prompt_number, :], 
            hashing_model.prompt_list[iteration][:init_prompt_number, :], 
            hashing_model.prompt_list[iteration][:init_prompt_number, :])
        prompt_feature_text = hashing_model.text_net.attention(
            hashing_model.prompt_list[iteration][init_prompt_number:, :], 
            hashing_model.prompt_list[iteration][init_prompt_number:, :], 
            hashing_model.prompt_list[iteration][init_prompt_number:, :])

        prompt_hash_code_image = hashing_model.image_net(torch.cat((prompt_feature_image[0], prompt_feature_image[0]), dim=1))
        prompt_hash_code_text = hashing_model.text_net(torch.cat((prompt_feature_text[0], prompt_feature_text[0]), dim=1))
    elif args.prompt_mode == 'share':
        prompt_feature_image = hashing_model.image_net.attention(
            hashing_model.prompt_list[iteration], 
            hashing_model.prompt_list[iteration], 
            hashing_model.prompt_list[iteration])
        prompt_feature_text = hashing_model.text_net.attention(
            hashing_model.prompt_list[iteration], 
            hashing_model.prompt_list[iteration],
            hashing_model.prompt_list[iteration])

        prompt_hash_code_image = hashing_model.image_net(torch.cat((prompt_feature_image[0], prompt_feature_image[0]), dim=1))
        prompt_hash_code_text = hashing_model.text_net(torch.cat((prompt_feature_text[0], prompt_feature_text[0]), dim=1))
    
    return torch.cat((prompt_hash_code_image, prompt_hash_code_text), dim=0)


def calc_map_k_unequal(qu_B, re_B, qu_L, re_L, topk=None):
    bits_len = re_B.shape[1]
    ex_bits_len = qu_B.shape[1] - bits_len
    num_query = qu_L.shape[0]
    map = 0
    if topk is None:
        topk = re_L.shape[0]
    for iter in range(num_query):
        q_L = qu_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)
        gnd = (q_L.mm(re_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        if ex_bits_len == 0:
            hamm = calc_hammingDist(qu_B[iter, :], re_B)
        elif ex_bits_len > 0:
            hamm = calc_hammingDist(qu_B[iter, :bits_len], re_B)

        _, ind = torch.sort(hamm, stable=True)   # 默认稳定排序
        ind.squeeze_()
        gnd = gnd[ind]
        tgnd = gnd[:topk]
        tsum = torch.sum(tgnd)
        if tsum == 0:
            continue
        count = torch.arange(1, int(tsum) + 1).type(torch.float32)
        tindex = torch.nonzero(tgnd).squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.cuda()
        map = map + torch.mean(count / tindex)
    map = map / num_query
    return map


def comput_mode_center_list(args, history_database_code_list_image, history_database_code_list_text):
    img_codes = torch.cat([c for c in history_database_code_list_image 
                          if c.shape[1] == history_database_code_list_image[-1].shape[1]])

    txt_codes = torch.cat([c for c in history_database_code_list_text 
                          if c.shape[1] == history_database_code_list_text[-1].shape[1]])

    image_center = compute_mode_center(img_codes)
    text_center = compute_mode_center(txt_codes)

    image_diff = (img_codes != image_center)         # [N, D]
    text_diff = (txt_codes != text_center)         # [N, D]
    image_hamming_distances = image_diff.sum(dim=1)   # [N]
    text_hamming_distances = text_diff.sum(dim=1)   # [N]

    img_over_threshold = (image_hamming_distances > (img_codes.shape[1] / 2)).sum().item()
    txt_over_threshold = (text_hamming_distances > (txt_codes.shape[1] / 2)).sum().item()

    if img_over_threshold > (img_codes.shape[0]*args.error_samples_ratio) or txt_over_threshold > (txt_codes.shape[0]*args.error_samples_ratio):
        extend_hash_length = True
    else:
        extend_hash_length = False
    args.extend_hash_length = extend_hash_length

    return image_center, text_center


def compute_mode_center(h_matrix):
    mode_center = torch.sign(torch.sum(h_matrix, dim=0))  # (hash_length,)
    mode_center_ste = h_matrix.mean(dim=0) + (mode_center - h_matrix.mean(dim=0)).detach()
    return mode_center_ste


def contrastive_loss(image_hash, label_matrix, temperature=0.1, epsilon=1e-8):
    sim_matrix = torch.matmul(image_hash, image_hash.T) / temperature
    label_overlap = torch.matmul(label_matrix.float(), label_matrix.float().T)  # [batch, batch]
    positive_mask = (label_overlap > 0).float()

    exp_sim = torch.exp(sim_matrix)
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))  # [batch, batch]
    
    numerator = (positive_mask * log_prob).sum(dim=1)
    denominator = positive_mask.sum(dim=1) + epsilon
    loss_per_sample = - numerator / denominator        # [batch]
    
    loss = loss_per_sample.mean()
    
    return loss
