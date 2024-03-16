import torch
from core.model_mutil import vit_gmvt_patchx_62_128 as create_model
import argparse


def count_parameters(model):
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")

    with open("parameters.txt", "w") as f:
        num_layers = 0
        num_neurons = 0
        num_conv_layers = 0
        num_fc_layers = 0
        num_residual_layers = 0
        num_dropout_layers = 0
        num_width = 0
        for name, param in model.named_parameters():
            f.write(f"{name}: {param.numel()}\n")
            print(f"{name}: {param.numel()}")
            num_layers += 1
            if "conv" in name:
                num_conv_layers += 1
            elif "fc" in name:
                num_fc_layers += 1
            if "residual" in name:
                num_residual_layers += 1
            if "dropout" in name:
                num_dropout_layers += 1
            if "weight" in name and len(param.shape) > 1:
                num_neurons += param.shape[0]
                num_width += param.shape[1]
        num_depth = num_layers - num_conv_layers - num_fc_layers - num_residual_layers - num_dropout_layers
        print(f"Number of layers: {num_layers}")
        print(f"Number of neurons: {num_neurons}")
        print(f"Number of convolutional layers: {num_conv_layers}")
        print(f"Number of fully connected layers: {num_fc_layers}")
        print(f"Number of residual layers: {num_residual_layers}")
        print(f"Number of dropout layers: {num_dropout_layers}")
        print(f"Model depth: {num_depth}")
        print(f"Model width: {num_width}")
        print(f"Model depth/width ratio: {num_depth / num_width:.3f}")


# 加载预训练模型
# model = models.resnet18(pretrained=True)

# device = torch.device(args.device if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
model = create_model(num_classes=9).to('cuda:0')
parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')  # 正常cuda：0就可以
state_dict = torch.load('..\\bestmodel\\model.pth')  # 使用本地磁盘上的模型参数文件
model.load_state_dict(state_dict)  # 把读入的模型参数加载到模型中

args = parser.parse_args()  # 解析参数
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
parser.add_argument('--weights', type=str,
                    default='.\\bestmodel\\model.pth',
                    help='initial weights path')

# 调用函数统计参数
count_parameters(model)
