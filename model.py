import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention

class BasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, *inputs):
        raise NotImplementedError("forward")
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path, map_location=None, strict=True):
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict, strict=strict)


class ImageModule(BasicModule):
    def __init__(self, args):
        super(ImageModule, self).__init__()
        self.args = args

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
    def __init__(self, args):
        super(HashingModel, self).__init__()
        self.args = args
        self.prompt_number = []
        self.prompt_list = nn.ParameterList()
        self.image_net = ImageModule(args)
        self.text_net = TextModule(args)

    def forward(self, image_feature, text_feature, iteration):
        if self.args.prompt_mode == 'specific':
            cosine_sim_image = torch.mm(image_feature, F.normalize(self.prompt_list[iteration][:int(self.prompt_number[iteration]/2), :].T))
            cosine_sim_text = torch.mm(text_feature, F.normalize(self.prompt_list[iteration][int(self.prompt_number[iteration]/2):, :].T))

            nearest_indices_image = torch.argmax(cosine_sim_image, dim=1)
            nearest_indices_text = torch.argmax(cosine_sim_text, dim=1)

            result_matrix_image = self.prompt_list[iteration][:int(self.prompt_number[iteration]/2), :][nearest_indices_image]
            result_matrix_text = self.prompt_list[iteration][int(self.prompt_number[iteration]/2):, :][nearest_indices_text]

        elif self.args.prompt_mode == 'share':
            cosine_sim_image = torch.mm(image_feature, self.prompt_list[iteration].T)
            cosine_sim_text = torch.mm(text_feature, self.prompt_list[iteration].T)

            nearest_indices_image = torch.argmax(cosine_sim_image, dim=1)
            nearest_indices_text = torch.argmax(cosine_sim_text, dim=1)

            result_matrix_image = self.prompt_list[iteration][nearest_indices_image]
            result_matrix_text = self.prompt_list[iteration][nearest_indices_text]

        enhanced_feature_image = self.image_net.attention(image_feature, result_matrix_image, result_matrix_image)
        enhanced_feature_text = self.text_net.attention(text_feature, result_matrix_text, result_matrix_text)

        final_feature_image = torch.cat((image_feature, enhanced_feature_image[0]), dim=1)
        final_feature_text = torch.cat((text_feature, enhanced_feature_text[0]), dim=1)

        disstill_feature_image = self.image_net(final_feature_text)
        disstill_feature_text = self.text_net(final_feature_image)

        image_hash = self.image_net(final_feature_image)
        text_hash = self.text_net(final_feature_text)
        
        return image_hash, text_hash, disstill_feature_image, disstill_feature_text

    @torch.no_grad()
    def test(self, image_feature, text_feature, iteration):
        if self.args.prompt_mode == 'specific':
            cosine_sim_image = torch.mm(image_feature, F.normalize(self.prompt_list[iteration][:int(self.prompt_number[iteration]/2), :].T))
            cosine_sim_text = torch.mm(text_feature, F.normalize(self.prompt_list[iteration][int(self.prompt_number[iteration]/2):, :].T))

            nearest_indices_image = torch.argmax(cosine_sim_image, dim=1)
            nearest_indices_text = torch.argmax(cosine_sim_text, dim=1)

            result_matrix_image = self.prompt_list[iteration][:int(self.prompt_number[iteration]/2), :][nearest_indices_image]
            result_matrix_text = self.prompt_list[iteration][int(self.prompt_number[iteration]/2):, :][nearest_indices_text]

        elif self.args.prompt_mode == 'share':
            cosine_sim_image = torch.mm(image_feature, self.prompt_list[iteration].T)
            cosine_sim_text = torch.mm(text_feature, self.prompt_list[iteration].T)

            nearest_indices_image = torch.argmax(cosine_sim_image, dim=1)
            nearest_indices_text = torch.argmax(cosine_sim_text, dim=1)

            result_matrix_image = self.prompt_list[iteration][nearest_indices_image]
            result_matrix_text = self.prompt_list[iteration][nearest_indices_text]

        enhanced_feature_image = self.image_net.attention(image_feature, result_matrix_image, result_matrix_image)
        enhanced_feature_text = self.text_net.attention(text_feature, result_matrix_text, result_matrix_text)

        final_feature_image = torch.cat((image_feature, enhanced_feature_image[0]), dim=1)
        final_feature_text = torch.cat((text_feature, enhanced_feature_text[0]), dim=1)

        image_hash = self.image_net(final_feature_image)
        text_hash = self.text_net(final_feature_text)
        
        return image_hash, text_hash

    def extend_model(self, extend_bit=0, extend_hash_length=False):

        self.image_net.extend(
            extend_bit=extend_bit,
            extend_hash_length=extend_hash_length
        )
        self.text_net.extend(
            extend_bit=extend_bit,
            extend_hash_length=extend_hash_length
        )

    def extract_prompt_hash_train(self):
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
        if args.prompt_mode == 'specific':
            self.prompt_number.append(int(init_prompt_number*2))
            prompt = nn.Parameter(
                torch.randn(init_prompt_number * 2, args.feature_dim, device="cuda"),
                requires_grad=True
            )
            self.prompt_list.append(prompt)
        elif args.prompt_mode == 'share':
            self.prompt_number.append(int(init_prompt_number))
            prompt = nn.Parameter(
                torch.randn(init_prompt_number, args.feature_dim, device="cuda"),
                requires_grad=True
            )
            self.prompt_list.append(prompt)
