import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import math
import argparse
from pathlib import Path
from model import HashingModel
from train import train_model
from load_data import get_loader
from utils import logger, set_seed, creat_result_dict, save_result_dict

parser = argparse.ArgumentParser()
# runid
parser.add_argument('--runid', type=str, default='1', help='run id')

parser.add_argument("--data_path", default="/path/to/data/mat/", type=str)
parser.add_argument("--output_dir", default="./outputs/", type=str)
parser.add_argument("--epoch", default=100, type=int)
parser.add_argument('--valid', type=bool, default=True, help='Whether to valid after per train.')
parser.add_argument('--valid_epoch', type=int, default=1, help='Number of epochs to valid.')
parser.add_argument('--seed', type=int, default=2025)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--feature_dim', type=int, default=512, help='dim of feature.')
parser.add_argument('--num_tasks', type=int, default=5, help='number of tasks')
parser.add_argument('--hamming_dis_threshold', type=float, default=0.5)
parser.add_argument('--extend_hash_length', type=bool, default=False, help='Whether to extend the hash length.')

parser.add_argument("--dataset_name", default="MSCOCO", type=str, help="MSCOCO/NUSWIDE")
parser.add_argument("--bit", default=16, type=int, help="16/32/64/128/256")
parser.add_argument("--prompt_mode", default='share', type=str, help="0: share, 1: separate")
parser.add_argument('--prompt_extend_per_sample_number', type=int, default=400)
parser.add_argument('--extend_bit', type=int, default=1)

# loss
parser.add_argument('--hash_cos_sim_loss', type=float, default=0.1)
parser.add_argument('--similartity_main_loss', type=float, default=100)
parser.add_argument('--binary_similartity_main_loss', type=float, default=1)
parser.add_argument('--quantify_loss', type=float, default=10)
parser.add_argument('--contras_loss', type=float, default=1)
parser.add_argument('--disstill_loss', type=float, default=200)
parser.add_argument('--prompt_hash_main_loss', type=float, default=0.1)
parser.add_argument('--radius_constraint_loss', type=float, default=10)

parser.add_argument('--learning_rate', type=float, default=0.00001)
parser.add_argument('--extend_learning_rate', type=float, default=0.00001)
parser.add_argument('--mothod', type=str, default='1')
parser.add_argument('--error_samples_ratio', type=float, default=0.1)

args = parser.parse_args()
args.extend_learning_rate = args.learning_rate * 0.1

def main():
    log = logger(args)
    set_seed(args.seed)
    args.dataset_path = os.path.join(args.data_path, args.dataset_name)

    checkpoint_folder = os.path.join(args.output_dir, 'checkpoints', '{}'.format(args.runid))
    csv_folder = os.path.join(args.output_dir, 'csv_result', '{}'.format(args.runid))

    Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)
    Path(csv_folder).mkdir(parents=True, exist_ok=True)

    checkpoints_path = checkpoint_folder + '/' + '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pth'.format(
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
    result_dict = creat_result_dict(args)

    prompt_hash_list = []
    history_database_code_list_image = []
    history_database_code_list_text = []
    # history_database_relax_code_list_image = []
    # history_database_relax_code_list_text = []
    mode_center_image, mode_center_text = None, None
    args.current_bit = args.bit

    hashing_model = HashingModel(args).cuda()
    
    for task_index in range(args.num_tasks):
        input_data_par, dataloader = get_loader(args, task_index=task_index)

        init_prompt_number = math.ceil(input_data_par['train_image'].shape[0] / args.prompt_extend_per_sample_number)
        log.info(f"Initialize the number of prompts: {init_prompt_number}")

        mode_center_image, mode_center_text = train_model(args, log, 
                    hashing_model, 
                    input_data_par, dataloader, 
                    init_prompt_number, task_index, 
                    result_dict, checkpoints_path, 
                    mode_center_image, mode_center_text, 
                    history_database_code_list_image, history_database_code_list_text, 
                    # history_database_relax_code_list_image, history_database_relax_code_list_text,
                    prompt_hash_list)
        log.info(f'The {task_index+1} task is trained')

    save_result_dict(args, result_dict, csv_folder)

if __name__ == '__main__':
    main()
