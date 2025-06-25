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
    log_path = log_floder_path + '/' + '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.log'.format(
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
        args.radius_constraint_loss, 
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
    csv_file = csv_folder + '/' + '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv'.format(
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
        args.radius_constraint_loss, 
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


def calc_neighbor(label1, label2, device):
    # calculate the similar matrix
    Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.FloatTensor)
    Sim = Sim.to(device)
    return Sim


def compute_mode_center(h_matrix):
    """计算哈希码的众数中心点（STE近似）
    
    Args:
        h_matrix (Tensor): 哈希码矩阵 (N, L)
        
    Returns:
        Tensor: 众数中心点 (L,)
        
    Note:
        使用直通估计器（STE）保持梯度流通
    """
    # 前向传播：计算众数
    # 计算每个比特位的众数（不可导）
    mode_center = torch.sign(torch.sum(h_matrix, dim=0))  # (hash_length,)
    # STE: 前向传播用众数，反向传播用原始矩阵的梯度
    mode_center_ste = h_matrix.mean(dim=0) + (mode_center - h_matrix.mean(dim=0)).detach()
    return mode_center_ste


def CenterSeparationLoss(margin, history_code_mode_center, now_code):
    """
    中心分离损失 - 防止当前任务哈希码与历史任务中心过于相似
    
    Args:
        margin (float): 间隔阈值，控制中心点最小间距
        history_code_mode_center (Tensor): 历史任务哈希码众数中心，形状(hash_length,)
        now_code (Tensor): 当前批次哈希码，形状(batch_size, hash_length)
    
    Returns:
        Tensor: 标量损失值，当相似度超过margin时产生惩罚
    
    Algorithm:
        1. 计算当前批次哈希码的众数中心
        2. 计算历史中心与当前中心的余弦相似度
        3. 当相似度超过margin时，用ReLU激活产生损失
    """
    # 计算当前批次中心（STE近似）
    c2 = compute_mode_center(now_code)  # (hash_length,)
    
    # 计算内积并归一化
    similarity = torch.dot(history_code_mode_center, c2) / now_code.size(1)  # 归一化到[-1, 1]
    
    # 计算间隔损失
    return F.relu(similarity - margin)


@torch.no_grad()
def extract_prompt_hash_test(args, hashing_model, iteration, init_prompt_number):
    """提取prompt的哈希编码
    
    Args:
        args (argparse): 配置参数（含prompt_mode）
        hashing_model (HashingModel): 哈希模型
        prompt_list (list): 当前所有prompt列表
        iteration (int): 当前任务迭代次数
        init_prompt_number (int): 初始prompt数量
        
    Returns:
        Tensor: 拼接后的prompt哈希码
        
    Note:
        根据prompt_mode选择专用/共享处理模式
    """
    if args.prompt_mode == 'specific':
        prompt_feature_image = hashing_model.image_net.attention(
            hashing_model.prompt_list[iteration][:init_prompt_number, :], 
            hashing_model.prompt_list[iteration][:init_prompt_number, :], 
            hashing_model.prompt_list[iteration][:init_prompt_number, :])
        prompt_feature_text = hashing_model.text_net.attention(
            hashing_model.prompt_list[iteration][init_prompt_number:, :], 
            hashing_model.prompt_list[iteration][init_prompt_number:, :], 
            hashing_model.prompt_list[iteration][init_prompt_number:, :])

        # fusion_prompt_image, fusion_prompt_text = hashing_model.fusion()

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

        # fusion_prompt_image, fusion_prompt_text = hashing_model.fusion()

        prompt_hash_code_image = hashing_model.image_net(torch.cat((prompt_feature_image[0], prompt_feature_image[0]), dim=1))
        prompt_hash_code_text = hashing_model.text_net(torch.cat((prompt_feature_text[0], prompt_feature_text[0]), dim=1))
    
    return torch.cat((prompt_hash_code_image, prompt_hash_code_text), dim=0)


def calc_map_k_unequal(task_index, args, qu_B, re_B, qu_L, re_L, topk=None):
    """
    不等长哈希码的mAP计算（支持多种对齐策略）
    
    Args:
        args.method: 处理策略选择（'1'-'7'）
        ex_bits_len: 查询码与库码的比特差（qu_bit - re_bit）
    
    Special Handling:
        当ex_bits_len < 0时（查询码比库码短）：
            - 方法1: 截断查询码
            - 方法2: 库码补零扩展
            - 方法3: 加权组合不同部分
            - 方法6: 滑动窗口找最小距离
    """
    bits_len = re_B.shape[1]
    ex_bits_len = qu_B.shape[1] - bits_len
    num_query = qu_L.shape[0]
    num_database = re_B.shape[0]
    map = 0
    if topk is None:
        topk = re_L.shape[0]
    for iter in range(num_query):
        q_L = qu_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)
        gnd = (q_L.mm(re_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        # 每个任务结束后，不等长哈希码的检索性能计算
        # 使用以下方法
        # -1- 汪博不等长哈希码检索（无需归一化的情况下就是截断）
        # -2- 短哈希码补零
        # -3- 短哈希码补零（权重）
        # -4- 序列匹配与编辑距离方法（实现过程有些复杂，浪费计算资源，舍弃）
        # -5- 动态规划对齐方法
        # -6- 滑动窗口匹配方法
        # -7- 基于核函数的方法
        if ex_bits_len == 0:
            # 直接计算
            hamm = calc_hammingDist(qu_B[iter, :], re_B)
        elif ex_bits_len > 0:
            if args.mothod == '1':
                # 策略1：截断查询码前bits_len位
                hamm = calc_hammingDist(qu_B[iter, :bits_len], re_B)
            elif args.mothod == '2':
                # 策略2：库码补零至查询码长度
                zero_matrix = torch.zeros((num_database, ex_bits_len)).cuda()
                re_B_extend = torch.cat((re_B, zero_matrix), dim=1)
                hamm = calc_hammingDist(qu_B[iter, :], re_B_extend)
            elif args.mothod == '3':
                # 策略3：分段加权组合
                zero_matrix = torch.zeros((num_database, ex_bits_len)).cuda()
                re_B_extend = torch.cat((re_B, zero_matrix), dim=1)
                hamm_1 = calc_hammingDist(qu_B[iter, :bits_len], re_B_extend[:, :bits_len])
                hamm_2 = calc_hammingDist(qu_B[iter, bits_len:], re_B_extend[:, bits_len:])
                hamm = hamm_1 + args.hamming_dis_threshold * hamm_2
            elif args.mothod == '4':
                # 策略6：滑动窗口找最小距离
                query_subcodes = sliding_windows(qu_B[iter, :], window_size=bits_len)
                min_dists = torch.full((num_database,), float('inf')).cuda()  # 初始化最小距离
                for subcode in query_subcodes:
                    # 计算当前子段与所有数据库项的汉明距离
                    dist = calc_hammingDist(subcode.view(1,-1), re_B)  # 输入维度必须为2D
                    # 保留每个数据库项的最大距离
                    min_dists = torch.maximum(min_dists, dist.squeeze())
                    # 保留每个数据库项的最小距离
                    # min_dists = torch.minimum(min_dists, dist.squeeze())
                hamm = min_dists
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


def sliding_windows(query, window_size=32):
    """生成滑动窗口子序列
    
    Args:
        query (Tensor): 输入序列 (L,)
        window_size (int): 窗口大小
        
    Returns:
        list[Tensor]: 窗口视图列表
    """
    return [query[i:i+window_size] for i in range(len(query)-window_size+1)]


def comput_mode_center_list(args, history_database_code_list_image, history_database_code_list_text):
    """计算历史哈希码的众数中心点
    
    Args:
        history_database_code_list_image (list[Tensor]): 图像哈希码列表
        history_database_code_list_text (list[Tensor]): 文本哈希码列表
        
    Returns:
        tuple: (图像中心点, 文本中心点)
    """
    # 合并图像哈希码
    img_codes = torch.cat([c for c in history_database_code_list_image 
                          if c.shape[1] == history_database_code_list_image[-1].shape[1]])
    
    # 合并文本哈希码
    txt_codes = torch.cat([c for c in history_database_code_list_text 
                          if c.shape[1] == history_database_code_list_text[-1].shape[1]])
    
    # 计算众数中心点
    image_center = compute_mode_center(img_codes)
    text_center = compute_mode_center(txt_codes)

    # 计算最大半径
    image_diff = (img_codes != image_center)         # [N, D]
    text_diff = (txt_codes != text_center)         # [N, D]
    image_hamming_distances = image_diff.sum(dim=1)   # [N]
    text_hamming_distances = text_diff.sum(dim=1)   # [N]
    # 计算超过哈希长度一半的样本数量
    img_over_threshold = (image_hamming_distances > (img_codes.shape[1] / 2)).sum().item()
    txt_over_threshold = (text_hamming_distances > (txt_codes.shape[1] / 2)).sum().item()

    if img_over_threshold > (img_codes.shape[0]*args.error_samples_ratio) or txt_over_threshold > (txt_codes.shape[0]*args.error_samples_ratio):
        extend_hash_length = True
    else:
        extend_hash_length = False
    args.extend_hash_length = extend_hash_length
    # image_radius = image_hamming_distances.max().item()
    # text_radius = text_hamming_distances.max().item()
    # if image_radius > (history_database_code_list_image[-1].shape[1])/2 or text_radius > (history_database_code_list_text[-1].shape[1])/2:
    #     extend_hash_length = True
    # else:
    #     extend_hash_length = False

    return image_center, text_center


def compute_mode_center(h_matrix):
    """计算哈希码的众数中心点（STE近似）
    
    Args:
        h_matrix (Tensor): 哈希码矩阵 (N, L)
        
    Returns:
        Tensor: 众数中心点 (L,)
        
    Note:
        使用直通估计器（STE）保持梯度流通
    """
    # 前向传播：计算众数
    # 计算每个比特位的众数（不可导）
    mode_center = torch.sign(torch.sum(h_matrix, dim=0))  # (hash_length,)
    # STE: 前向传播用众数，反向传播用原始矩阵的梯度
    mode_center_ste = h_matrix.mean(dim=0) + (mode_center - h_matrix.mean(dim=0)).detach()
    return mode_center_ste


import torch
import torch.nn.functional as F

def contrastive_loss(image_hash, label_matrix, temperature=0.1, epsilon=1e-8):
    """
    image_hash: 图像哈希码 [batch_size, hash_dim]
    label_matrix: 标签矩阵 [batch_size, num_classes], 多标签情况下为0/1矩阵
    temperature: 温度系数
    epsilon: 防止除零的小常数
    """
    # 2. 计算特征相似度矩阵 [batch, batch]
    sim_matrix = torch.matmul(image_hash, image_hash.T) / temperature
    
    # 3. 根据标签矩阵构建正样本掩码 [batch, batch]
    #    多标签情况下，两个样本共享至少一个类别则视为正样本
    label_overlap = torch.matmul(label_matrix.float(), label_matrix.float().T)  # [batch, batch]
    positive_mask = (label_overlap > 0).float()  # 共享至少一个类别则为1
    
    # 4. 排除自身对角线（可选，根据需求决定是否保留自身作为正样本）
    # positive_mask.fill_diagonal_(0)  # 如果不需要自身作为正样本则取消注释
    
    # 5. 计算对比损失（多标签适配版）
    # 5.1 计算正样本对数概率
    exp_sim = torch.exp(sim_matrix)
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))  # [batch, batch]
    
    # 5.2 计算每个样本的正样本平均损失
    numerator = (positive_mask * log_prob).sum(dim=1)  # 分子：正样本对数概率和
    denominator = positive_mask.sum(dim=1) + epsilon   # 分母：正样本数量（防除零）
    loss_per_sample = - numerator / denominator        # [batch]
    
    # 5.3 最终损失取批次平均
    loss = loss_per_sample.mean()
    
    return loss

        
def MultiLabelContrastiveLoss(image_emb, text_emb, label_sim):
    # 计算相似度矩阵
    logits = torch.mm(image_emb, text_emb.T) / 0.1  # [N, N]
    
    # 正样本权重（共享标签数）
    pos_weight = label_sim  
    
    # 负样本权重（归一化处理）
    neg_weight = 1 - pos_weight
    
    # 计算对比损失
    exp_logits = torch.exp(logits)
    pos_term = (pos_weight * logits).sum(dim=1)            # 分子项
    neg_term = torch.log(exp_logits * neg_weight).sum(dim=1) # 分母项
    
    loss = - (pos_term - neg_term).mean()
    return loss