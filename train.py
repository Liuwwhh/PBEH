import os
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from load_data import get_data
from utils import calc_map_k, calc_map_k_unequal, CenterSeparationLoss, extract_prompt_hash_test, comput_mode_center_list, contrastive_loss

loss_l2 = torch.nn.MSELoss()
criterion = nn.CosineSimilarity(dim=1)

def train_model(args, log, 
                hashing_model, 
                input_data_par, dataloader, 
                init_prompt_number, task_index, 
                result_dict, checkpoints_path, 
                mode_center_image, mode_center_text, 
                history_database_code_list_image, history_database_code_list_text, 
                # history_database_relax_code_list_image, history_database_relax_code_list_text,
                prompt_hash_list):
    # If it's not the first task then the previous model needs to be loaded
    if task_index != 0:
        hashing_model.load(checkpoints_path)
        hashing_model.cuda()
    
    # Initialize the hash layer
    hashing_model.eval()
    if args.extend_hash_length:
        args.current_bit += args.extend_bit
        hashing_model.extend_model(extend_bit=args.extend_bit, extend_hash_length=args.extend_hash_length)
        log.info(f'extend_hash_length: {args.extend_bit}')
    hashing_model.cuda()

    torch.manual_seed(args.seed)
    hashing_model.add_prompt(args, init_prompt_number)
    # optimizer
    # 剔除掉不需要训练的参数
    if task_index == 0:
        optimizer = optim.AdamW(
            [
                {'params': hashing_model.image_net.parameters(), 'lr': args.learning_rate},
                {'params': hashing_model.text_net.parameters(), 'lr': args.learning_rate},
                {'params': hashing_model.prompt_list[-1], 'lr': args.learning_rate},
            ]
        )
    else:
        optimizer = optim.AdamW(
            [
                {'params': hashing_model.image_net.parameters(), 'lr': args.extend_learning_rate},
                {'params': hashing_model.text_net.parameters(), 'lr': args.extend_learning_rate},
                {'params': hashing_model.prompt_list[-1], 'lr': args.learning_rate},
            ]
        )
    # optimizer = optim.AdamW(hashing_model.parameters(), lr=learning_rate)

    max_mapi2t = max_mapt2i = torch.zeros(1, dtype=torch.float32)
    max_mapi2t = max_mapi2t.cuda()
    max_mapt2i = max_mapt2i.cuda()
    train_dataloader = dataloader['train']
    for epoch in tqdm(range(args.epoch)):
        for batch in train_dataloader:
            image, text, label = batch
            image = image.cuda()
            text = text.cuda()
            label = label.cuda()
            hashing_model.train()
            optimizer.zero_grad()

            # get hash code
            image_hash, text_hash, disstill_feature_image, disstill_feature_text = hashing_model(image, text, task_index)
            hash_similarity = image_hash.mm(text_hash.t())
            binary_hash_similarity = torch.sign(image_hash).mm(torch.sign(text_hash).t())
            similarity_matrix = label.mm(label.t()) / 16.0
            similarity_matrix = 1 / (1+torch.exp(-similarity_matrix))
            similarity_matrix = 2 * similarity_matrix.float() - 1

            # 损失计算 ----------------------------------------------------------
            # 松弛码余弦相似
            hash_cos_sim_loss = (torch.sum(1 - criterion(image_hash, text_hash)) + torch.sum(1 - criterion(torch.sign(image_hash), torch.sign(text_hash)))) * args.hash_cos_sim_loss


            # 相似度保持
            similartity_main_loss = (loss_l2(hash_similarity, similarity_matrix) + loss_l2(hash_similarity.t(), similarity_matrix)) * args.similartity_main_loss
            binary_similartity_main_loss = (loss_l2(binary_hash_similarity, similarity_matrix) + loss_l2(binary_hash_similarity.t(), similarity_matrix)) * args.binary_similartity_main_loss

            # 量化损失
            quantify_loss = loss_l2(image_hash, torch.sign(image_hash)) + loss_l2(text_hash, torch.sign(text_hash))
            quantify_loss = quantify_loss * args.quantify_loss

            # 判别性损失
            contras_loss = (contrastive_loss(image_hash, label) + contrastive_loss(text_hash, label)) * args.contras_loss

            # 蒸馏损失
            disstill_loss = (loss_l2(image_hash, disstill_feature_text) + loss_l2(text_hash, disstill_feature_image)) * args.disstill_loss

            # 总损失
            loss = hash_cos_sim_loss\
                + similartity_main_loss + binary_similartity_main_loss\
                      + quantify_loss + contras_loss\
                          + disstill_loss
            # 损失记录 log.info
            # 新任务学习特有损失 --------------------------------------------------
            # 中心分离约束（防止新任务哈希码与历史中心过近）
            if task_index == 0:
                pass
            else:
                if args.extend_hash_length:
                    pass
                else:
                    # 图像模态中心约束
                    radius_constraint_loss_image = CenterSeparationLoss(
                        margin=-1, 
                        history_code_mode_center=mode_center_image, 
                        now_code=torch.sign(image_hash[:, :mode_center_image.shape[0]])
                    )
                    # 文本模态中心约束
                    radius_constraint_loss_text = CenterSeparationLoss(
                        margin=-1, 
                        history_code_mode_center=mode_center_text, 
                        now_code=torch.sign(text_hash[:, :mode_center_text.shape[0]])
                    )
                    loss += (radius_constraint_loss_image + radius_constraint_loss_text) * args.radius_constraint_loss
                # 提示哈希一致性约束 -------------------------------------------------
                # 提取当前提示哈希码
                prompt_hash_main_loss = 0.
                history_prompt_hash_list = hashing_model.extract_prompt_hash_train()
                for prompt_hash_index in range(task_index):
                    prompt_hash = prompt_hash_list[prompt_hash_index]
                    history_prompt_hash = history_prompt_hash_list[prompt_hash_index][:, :prompt_hash.shape[1]]
                    prompt_hash_main_loss += torch.abs(prompt_hash - history_prompt_hash).sum()
                loss += prompt_hash_main_loss * args.prompt_hash_main_loss

            # 优化
            # log.info(f"all_loss {loss}")
            loss.backward()
            optimizer.step()

        if epoch % args.valid_epoch == 0:
            if args.valid:
                hashing_model.eval()
                with torch.no_grad():
                    separation = True
                    mapi2t_1000, mapt2i_1000 = valid_fun(hashing_model, args, 
                                                input_data_par['test_image'], input_data_par['database_image'],
                                                input_data_par['test_text'], input_data_par['database_text'],
                                                input_data_par['test_label'], input_data_par['database_label'],
                                                task_index, separation, top_k=1000)
                log.info('...epoch: %3d, valid MAP: MAP_1000(i->t): %3.4f, MAP_1000(t->i): %3.4f' % (epoch + 1, mapi2t_1000, mapt2i_1000))
                if mapi2t_1000+mapt2i_1000 > max_mapi2t+max_mapt2i:
                    max_mapi2t = mapi2t_1000
                    max_mapt2i = mapt2i_1000
                    hashing_model.save(checkpoints_path)
        hashing_model.train()

    result_dict['I2T'][task_index, task_index] = max_mapi2t.cpu().numpy()
    result_dict['T2I'][task_index, task_index] = max_mapt2i.cpu().numpy()
    log.info('...test MAP: MAP_1000(i->t): %3.4f, MAP_1000(t->i): %3.4f' % (max_mapi2t, max_mapt2i))

    log.info(f'Start evaluation!')
    hashing_model.load(checkpoints_path)
    hashing_model.cuda()
    hashing_model.eval()

    with torch.no_grad():
        rBX, rBY = generate_code(hashing_model, input_data_par['database_image'], input_data_par['database_text'], args, task_index)
        prompt_hash_list.append(extract_prompt_hash_test(args, hashing_model, task_index, init_prompt_number))

    history_database_code_list_image.append(rBX)
    history_database_code_list_text.append(rBY)

    # 所有任务都要进行扩展，取消中心约束
    mode_center_image, mode_center_text = comput_mode_center_list(args, history_database_code_list_image, history_database_code_list_text)

    if task_index != 0:
        for i in range(task_index+1):
            _, _, _, test_image, test_text, test_label, _, _, database_label = get_data(args.dataset_path, i, args.dataset_name)
            test_image = test_image.cuda()
            test_text = test_text.cuda()
            test_label = test_label.cuda()
            database_label = database_label.cuda()

            qBX, qBY = generate_code(hashing_model, test_image, test_text, args, task_index)

            mapi2t_1000 = calc_map_k_unequal(i, args, qBX, history_database_code_list_text[i], test_label, database_label, 1000)
            mapt2i_1000 = calc_map_k_unequal(i, args, qBY, history_database_code_list_image[i], test_label, database_label, 1000)
            log.info('...The {} data test is finished...'.format(i+1))
            log.info('...test MAP: MAP_1000(i->t): %3.4f, MAP_1000(t->i): %3.4f' % (mapi2t_1000, mapt2i_1000))
            if task_index == 0:
                result_dict['I2T'][task_index, args.num_tasks] = mapi2t_1000.cpu().numpy()
                result_dict['T2I'][task_index, args.num_tasks] = mapt2i_1000.cpu().numpy()
            result_dict['I2T'][task_index, i] = mapi2t_1000.cpu().numpy()
            result_dict['T2I'][task_index, i] = mapt2i_1000.cpu().numpy()

    return mode_center_image, mode_center_text


def valid_fun(hashing_model, args, query_x, retrieval_x, query_y, retrieval_y, query_L, retrieval_L, task_index, separation, top_k):
    qBX, qBY = generate_code(hashing_model, query_x, query_y, args, task_index)
    rBX, rBY = generate_code(hashing_model, retrieval_x, retrieval_y, args, task_index)

    mapi2t_1000 = calc_map_k(qBX, rBY, query_L, retrieval_L, top_k)
    mapt2i_1000 = calc_map_k(qBY, rBX, query_L, retrieval_L, top_k)
    # mapi2t_1000_I2I = calc_map_k(qBX, rBX, query_L, retrieval_L, top_k)
    # mapt2i_1000_T2T = calc_map_k(qBY, rBY, query_L, retrieval_L, top_k)
    # print('mapi2t_1000:', mapi2t_1000_I2I)
    # print('mapt2i_1000:', mapt2i_1000_T2T)
    return mapi2t_1000, mapt2i_1000

@torch.no_grad()
def generate_code(hashing_model, X, Y, args, task_index):
    num_data = X.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B_X = torch.zeros(num_data, args.current_bit, dtype=torch.float).cuda()
    B_Y = torch.zeros(num_data, args.current_bit, dtype=torch.float).cuda()

    for i in range(num_data // args.batch_size + 1):
        ind = index[i * args.batch_size: min((i + 1) * args.batch_size, num_data)]
        image = X[ind].type(torch.float).cuda()
        text = Y[ind].type(torch.float).cuda()
        X_hash, Y_hash = hashing_model.test(image, text, task_index)
        B_X[ind] = X_hash
        B_Y[ind] = Y_hash
    B_X = torch.sign(B_X)
    B_Y = torch.sign(B_Y)
    return B_X, B_Y