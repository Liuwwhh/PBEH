import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention

class BasicModule(nn.Module):
    """神经网络模块基类，提供通用功能
    
    功能特性：
    - 定义标准前向传播接口
    - 封装模型参数保存/加载逻辑
    - 强制子类实现forward方法
    
    子类需实现：
    - forward(): 定义前向传播逻辑
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, *inputs):
        """定义前向传播逻辑（子类必须实现）"""
        raise NotImplementedError("所有子类必须实现forward方法")
    
    def save(self, path):
        """安全保存模型参数到指定路径
        
        Args:
            path (str): 模型保存路径（建议.pt或.pth扩展名）
        """
        torch.save(self.state_dict(), path)
    
    def load(self, path, map_location=None, strict=True):
        """从文件加载模型参数
        
        Args:
            path (str): 模型文件路径
            map_location (device): 指定加载设备（如cpu/cuda:0）
            strict (bool): 是否严格匹配参数名称
        """
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict, strict=strict)


class ImageModule(BasicModule):
    def __init__(self, args):
        super(ImageModule, self).__init__()
        self.args = args

        # 原始网络结构配置 --------------------------------------------
        self.original_structure = {
            'Classifier': {'in': 4096, 'out': args.bit}
        }
        self.new_structure = {}

        self.rule1 = nn.ReLU(inplace=True)
        self.rule2 = nn.ReLU(inplace=True)
        self.attention = MultiheadAttention(embed_dim=512, num_heads=8)
        # fc1
        self.Linear1 = nn.Linear(in_features=2*args.feature_dim, out_features=4096)
        # fc2
        self.Linear2 = nn.Linear(in_features=4096, out_features=4096)
        # fc3
        self.Classifier = nn.Linear(in_features=4096, out_features=args.bit)

    @torch.no_grad()
    def extend(self, extend_bit=0, extend_hash_length=False):
        # 扩展分类器
        original_in = self.original_structure['Classifier']['in']
        original_out = self.original_structure['Classifier']['out']
        new_in = 4096
        if extend_hash_length:
            new_out = original_out + extend_bit
        else:
            new_out = original_out
        weight_data_copy = self.Classifier.weight.data.clone().detach()
        bias_data_copy = self.Classifier.bias.data.clone().detach()
        new_Classifier = nn.Linear(in_features=new_in, out_features=new_out)
        new_Classifier.weight.data[:original_out, :original_in] = weight_data_copy
        new_Classifier.bias.data[:original_out] = bias_data_copy
        setattr(self, 'Classifier', new_Classifier)
        self.new_structure['Classifier'] = {'in': new_in, 'out': new_out}
        self.original_structure = self.new_structure

    def forward(self, x):
        out_Linear1 = self.Linear1(x)
        out_Linear1 = self.rule1(out_Linear1)
        out_Linear2 = self.Linear2(out_Linear1)
        out_Linear2 = self.rule2(out_Linear2)
        out_Classifier = self.Classifier(out_Linear2)
        out = F.normalize(out_Classifier)
        return out
        

class TextModule(BasicModule):
    def __init__(self, args):
        super(TextModule, self).__init__()
        self.args = args

        # 原始网络结构配置 --------------------------------------------
        self.original_structure = {
            'Classifier': {'in': 4096, 'out': args.bit}
        }
        self.new_structure = {}

        self.rule1 = nn.ReLU(inplace=True)
        self.rule2 = nn.ReLU(inplace=True)
        self.attention = MultiheadAttention(embed_dim=512, num_heads=8)
        # fc1
        self.Linear1 = nn.Linear(in_features=2*args.feature_dim, out_features=4096)
        # fc2
        self.Linear2 = nn.Linear(in_features=4096, out_features=4096)
        # fc3
        self.Classifier = nn.Linear(in_features=4096, out_features=args.bit)

    @torch.no_grad()
    def extend(self, extend_bit=0, extend_hash_length=False):
        # 扩展分类器
        original_in = self.original_structure['Classifier']['in']
        original_out = self.original_structure['Classifier']['out']
        new_in = 4096
        if extend_hash_length:
            new_out = original_out + extend_bit
        else:
            new_out = original_out
        weight_data_copy = self.Classifier.weight.data.clone().detach()
        bias_data_copy = self.Classifier.bias.data.clone().detach()
        new_Classifier = nn.Linear(in_features=new_in, out_features=new_out)
        new_Classifier.weight.data[:original_out, :original_in] = weight_data_copy
        new_Classifier.bias.data[:original_out] = bias_data_copy
        setattr(self, 'Classifier', new_Classifier)
        self.new_structure['Classifier'] = {'in': new_in, 'out': new_out}
        self.original_structure = self.new_structure

    def forward(self, x):
        out_Linear1 = self.Linear1(x)
        out_Linear1 = self.rule1(out_Linear1)
        out_Linear2 = self.Linear2(out_Linear1)
        out_Linear2 = self.rule2(out_Linear2)
        out_Classifier = self.Classifier(out_Linear2)
        out = F.normalize(out_Classifier)
        return out
    

class HashingModel(BasicModule):
    """多模态哈希模型核心类，实现提示驱动的特征增强与哈希编码
    
    特性：
    - 支持特定模态/共享提示两种模式
    - 动态扩展模型容量
    - 多粒度特征融合机制
    
    Args:
        args (argparse): 配置参数对象，需包含：
            - prompt_mode (str): 提示模式（'specific'/'share'）
    """
    def __init__(self, args):
        super(HashingModel, self).__init__()
        self.args = args
        # 提示数量记录（用于动态扩展）
        self.prompt_number = []
        self.prompt_list = nn.ParameterList()
        # 初始化图像和文本编码器
        self.image_net = ImageModule(args)
        self.text_net = TextModule(args)

    def forward(self, image_feature, text_feature, iteration):
        """多模态前向传播流程
        
        Args:
            image_feature (Tensor): 图像特征 (bs, 512)
            text_feature (Tensor): 文本特征 (bs, 512)
            prompt_list (list[Tensor]): 多任务提示集合
            iteration (int): 当前任务迭代次数
            
        Returns:
            tuple: (图像哈希码, 文本哈希码)
            
        流程说明：
            1. 特征与提示归一化
            2. 基于余弦相似度的提示选择
            3. 注意力增强特征
            4. 跨任务提示融合
            5. 最终哈希编码
        """
        # === 提示选择逻辑 ===
        if self.args.prompt_mode == 'specific':
            # 计算余弦相似度矩阵
            cosine_sim_image = torch.mm(image_feature, F.normalize(self.prompt_list[iteration][:int(self.prompt_number[iteration]/2), :].T))
            cosine_sim_text = torch.mm(text_feature, F.normalize(self.prompt_list[iteration][int(self.prompt_number[iteration]/2):, :].T))

            # # 找到最相似索引
            nearest_indices_image = torch.argmax(cosine_sim_image, dim=1)
            nearest_indices_text = torch.argmax(cosine_sim_text, dim=1)

            # 提取结果矩阵
            result_matrix_image = self.prompt_list[iteration][:int(self.prompt_number[iteration]/2), :][nearest_indices_image]
            result_matrix_text = self.prompt_list[iteration][int(self.prompt_number[iteration]/2):, :][nearest_indices_text]

        elif self.args.prompt_mode == 'share':
            # 计算余弦相似度矩阵
            cosine_sim_image = torch.mm(image_feature, self.prompt_list[iteration].T)
            cosine_sim_text = torch.mm(text_feature, self.prompt_list[iteration].T)

            # 找到最相似索引
            nearest_indices_image = torch.argmax(cosine_sim_image, dim=1)
            nearest_indices_text = torch.argmax(cosine_sim_text, dim=1)

            # 提取结果矩阵
            result_matrix_image = self.prompt_list[iteration][nearest_indices_image]
            result_matrix_text = self.prompt_list[iteration][nearest_indices_text]

        # === 特征增强 ===
        enhanced_feature_image = self.image_net.attention(image_feature, result_matrix_image, result_matrix_image)
        enhanced_feature_text = self.text_net.attention(text_feature, result_matrix_text, result_matrix_text)

        # === 特征拼接 ===
        final_feature_image = torch.cat((image_feature, enhanced_feature_image[0]), dim=1)
        final_feature_text = torch.cat((text_feature, enhanced_feature_text[0]), dim=1)

        disstill_feature_image = self.image_net(final_feature_text)
        disstill_feature_text = self.text_net(final_feature_image)

        # === 哈希生成 ===
        image_hash = self.image_net(final_feature_image)
        text_hash = self.text_net(final_feature_text)
        
        return image_hash, text_hash, disstill_feature_image, disstill_feature_text

    @torch.no_grad()
    def test(self, image_feature, text_feature, iteration):
        """多模态前向传播流程
        
        Args:
            image_feature (Tensor): 图像特征 (bs, 512)
            text_feature (Tensor): 文本特征 (bs, 512)
            prompt_list (list[Tensor]): 多任务提示集合
            iteration (int): 当前任务迭代次数
            
        Returns:
            tuple: (图像哈希码, 文本哈希码)
            
        流程说明：
            1. 特征与提示归一化
            2. 基于余弦相似度的提示选择
            3. 注意力增强特征
            4. 跨任务提示融合
            5. 最终哈希编码
        """
        # === 提示选择逻辑 ===
        if self.args.prompt_mode == 'specific':
            # 计算余弦相似度矩阵
            cosine_sim_image = torch.mm(image_feature, F.normalize(self.prompt_list[iteration][:int(self.prompt_number[iteration]/2), :].T))
            cosine_sim_text = torch.mm(text_feature, F.normalize(self.prompt_list[iteration][int(self.prompt_number[iteration]/2):, :].T))

            # # 找到最相似索引
            nearest_indices_image = torch.argmax(cosine_sim_image, dim=1)
            nearest_indices_text = torch.argmax(cosine_sim_text, dim=1)

            # 提取结果矩阵
            result_matrix_image = self.prompt_list[iteration][:int(self.prompt_number[iteration]/2), :][nearest_indices_image]
            result_matrix_text = self.prompt_list[iteration][int(self.prompt_number[iteration]/2):, :][nearest_indices_text]

        elif self.args.prompt_mode == 'share':
            # 计算余弦相似度矩阵
            cosine_sim_image = torch.mm(image_feature, self.prompt_list[iteration].T)
            cosine_sim_text = torch.mm(text_feature, self.prompt_list[iteration].T)

            # 找到最相似索引
            nearest_indices_image = torch.argmax(cosine_sim_image, dim=1)
            nearest_indices_text = torch.argmax(cosine_sim_text, dim=1)

            # 提取结果矩阵
            result_matrix_image = self.prompt_list[iteration][nearest_indices_image]
            result_matrix_text = self.prompt_list[iteration][nearest_indices_text]

        # === 特征增强 ===
        enhanced_feature_image = self.image_net.attention(image_feature, result_matrix_image, result_matrix_image)
        enhanced_feature_text = self.text_net.attention(text_feature, result_matrix_text, result_matrix_text)

        # === 特征拼接 ===
        final_feature_image = torch.cat((image_feature, enhanced_feature_image[0]), dim=1)
        final_feature_text = torch.cat((text_feature, enhanced_feature_text[0]), dim=1)

        # === 哈希生成 ===
        image_hash = self.image_net(final_feature_image)
        text_hash = self.text_net(final_feature_text)
        
        return image_hash, text_hash

    def extend_model(self, extend_bit=0, extend_hash_length=False):
        """模型扩展入口
        
        Args:
            extend_mode (str): 扩展模式（'number'数值/'factor'比例）
            extend_value (int/float): 扩展量
            extend_bit (int): 哈希码扩展位数
            extend_hash_length (bool): 是否改变哈希码长度
        """
        self.image_net.extend(
            extend_bit=extend_bit,
            extend_hash_length=extend_hash_length
        )
        self.text_net.extend(
            extend_bit=extend_bit,
            extend_hash_length=extend_hash_length
        )

    def extract_prompt_hash_train(self):
        """提取提示特征对应的哈希码
        
        用于：
            - 跨任务约束
            - 提示特征可视化
            
        Returns:
            Tensor: 拼接后的提示哈希码 (num_prompts*2, bit)
        """
        if self.args.prompt_mode == 'specific':
            history_prompt_hash_code = []
            for prompt_index in range(len(self.prompt_list)-1):
                number = int(self.prompt_number[prompt_index]/2)
                prompt_feature_image = self.image_net.attention(self.prompt_list[prompt_index][:number, :], self.prompt_list[prompt_index][:number, :], self.prompt_list[prompt_index][:number, :])
                prompt_feature_text = self.text_net.attention(self.prompt_list[prompt_index][number:, :], self.prompt_list[prompt_index][number:, :], self.prompt_list[prompt_index][number:, :])

                prompt_hash_code_image = self.image_net(torch.cat((prompt_feature_image[0], prompt_feature_image[0]), dim=1))
                prompt_hash_code_text = self.text_net(torch.cat((prompt_feature_text[0], prompt_feature_text[0]), dim=1))
                history_prompt_hash_code.append(torch.cat((prompt_hash_code_image, prompt_hash_code_text), dim=0))
            return history_prompt_hash_code
        elif self.args.prompt_mode == 'share':
            history_prompt_hash_code = []
            for prompt_index in range(len(self.prompt_list)-1):
                prompt_feature_image = self.image_net.attention(self.prompt_list[prompt_index], self.prompt_list[prompt_index], self.prompt_list[prompt_index])
                prompt_feature_text = self.text_net.attention(self.prompt_list[prompt_index], self.prompt_list[prompt_index], self.prompt_list[prompt_index])

                prompt_hash_code_image = self.image_net(torch.cat((prompt_feature_image[0], prompt_feature_image[0]), dim=1))
                prompt_hash_code_text = self.text_net(torch.cat((prompt_feature_text[0], prompt_feature_text[0]), dim=1))
                history_prompt_hash_code.append(torch.cat((prompt_hash_code_image, prompt_hash_code_text), dim=0))
            return history_prompt_hash_code
    
    def add_prompt(self, args, init_prompt_number):
        # 根据提示模式初始化提示参数
        if args.prompt_mode == 'specific':  # 模态专属提示
            self.prompt_number.append(int(init_prompt_number*2))
            prompt = nn.Parameter(
                torch.randn(init_prompt_number * 2, args.feature_dim, device="cuda"),
                requires_grad=True
            )
            self.prompt_list.append(prompt)
        elif args.prompt_mode == 'share':   # 模态共享提示
            self.prompt_number.append(int(init_prompt_number))
            # 模态共享提示参数
            prompt = nn.Parameter(
                torch.randn(init_prompt_number, args.feature_dim, device="cuda"),
                requires_grad=True
            )
            self.prompt_list.append(prompt)