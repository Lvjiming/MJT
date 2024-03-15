import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functools import partial
from collections import OrderedDict


# 定义基本的残差块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class view_encoder(nn.Module):
    def __init__(self):
        super(view_encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x


# ---------------------------------------------------------------------------------------------------------------------
# `in_channels` in this code indicates the number of channels of the input X's shape.
# `out_channels` represents the dimension of the output features, i.e., the dimension of the features after processing
# by the self-attention mechanism. In this module, the input features are mapped to the `out_channels` dimensional
# feature space by three linear layers. Then the attention matrix is computed, and finally the feature representation
# after the self-attention mechanism is obtained by weighted summation.
# It can be adapted to the specific task. In this model, `out_channels` is a hyperparameter, which can be set according
# to the actual situation,
# and it is generally recommended to set it to the commonly used feature dimensions such as 64, 128, 256, and so on.
# ---------------------------------------------------------------------------------------------------------------------

class MultiViewAttention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.5,
                 proj_drop_ratio=0.5):
        super(MultiViewAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # scale is the calculation of the denominator of the ATTENTION formula sqrt(dk)
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)  # Splice each head and multiply it by a W0, this step is to achieve this
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # x's shape, here passed an image or a feature image extracted by
        # the neural network [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # .permute(2, 0, 3, 1, 4) Used to swap the internal order of vectors or arrays, a number that represents the
        # transfer of data from the original second dimension to dimension 0.
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        # represents matrix multiplication, in addition qkv is a multi-dimensional matrix, there is no way to directly
        # matrix multiplication, so used the transpose transformation, then only the last two dimensions of the data
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]，
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # Here multiplying V is a weighted summing operation. #
        x = self.proj(x)  #
        x = self.proj_drop(x)  # dropout features
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size_w=128, img_size_h=64, patch_size=16, in_c=512, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size_w, img_size_h)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.num_patches = 256
        # conv2d four metrics, number of channels, number of convolution kernels, convolution kernel size and step size
        # self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Problem with the following line, size mismatch for patch_embed.proj.weight: copying a param with shape
        # torch.Size([768, 3, 16, 16]) from checkpoint, the shape in current model is torch.Size([768, 512, 1, 1]).
        # The size of this convolution kernel is adjusted according to the image input size,
        # kernel_size=input_width - (target_length - step_length)
        self.proj1 = nn.Conv2d(in_c, embed_dim, kernel_size=(28 - 13, 15), stride=(1, 1))  #
        self.proj2 = nn.Conv2d(in_c, embed_dim, kernel_size=(56 - 13, 15), stride=(1, 1))  #
        self.proj3 = nn.Conv2d(in_c, embed_dim, kernel_size=(84 - 13, 15), stride=(1, 1))  #
        self.norm_layer = nn.BatchNorm1d(embed_dim)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        # def forward(self, x, num_view=1):

    def forward(self, x):
        # B, C, H, W = x.shape
        # num_patches = H * W + 1
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        # x = x.flatten(2).transpose(1, 2)
        # --------------------------------------------------------------------------------------------------------------
        # x is torch.Size([64, 128, 28, 28]) before spreading, two views; spreading, torch.Size([64, 14*14=196, 768])
        # self.proj1(x): this part represents a linear transformation operation by passing the input tensor x through
        # the self.proj1 linear transformation layer. This linear transformation can be a fully connected layer or
        # a convolutional layer, depending on your specific model.
        # .flatten(2): This section uses the .flatten(2) method to flatten the result of self.proj1(x).
        # Specifically, it combines the last two dimensions of the tensor (2 and 3 dimensions) into a single dimension,
        # leaving the first dimension unchanged. The purpose of this is to convert the matrix or tensor
        # into a shape suitable for transpose operations.transpose(1, 2): This section uses the .transpose(1, 2) method
        # to perform a transpose operation on the flattened tensor. Specifically, it swaps the positions of the first
        # and second dimensions, i.e., the original shape (batch_size, seq_len, feature_dim) is converted to
        # (batch_size, feature_dim, seq_len). This transpose operation is often used to move the dimension of the
        # sequence length to the last dimension for subsequent operations.
        # --------------------------------------------------------------------------------------------------------------
        x = self.proj1(x).flatten(2).transpose(1, 2)
        # if num_view == 1:
        #     x = self.proj1(x).flatten(2).transpose(1, 2)
        # elif num_view == 2:
        #     x = self.proj2(x).flatten(2).transpose(1, 2)
        # else:
        #     x = self.proj3(x).flatten(2).transpose(1, 2)
        # x = self.proj(x).flatten(2)
        # x = self.norm_layer(x).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.5,
                 proj_drop_ratio=0.5):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # denotes the image dimension of each head, and the floor divides the
        # This is the calculation of the denominator of the attentions formula sqrt(dk) #
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # x's shape, here passed an image or a feature image extracted by the neural network
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # .permute(2, 0, 3, 1, 4)用来调换向量或者数组的内部顺序，一个数字，表示把原来第二维的数据转到第0维
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torch script happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        # @这里表示矩阵乘法，另外qkv是个多维矩阵，没办法直接进行矩阵乘法，因此这里用了transpose变换，则只对最后两维数据进行操作
        # 这里的qkv操作就是针对多头中的每一个头进行操作的，self.scale是做一个normal处理，归一化，torch.Size([96, 12, 197, 197])
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # torch.Size([96, 12, 197, 197])
        attn = self.attn_drop(attn)  # torch.Size([96, 12, 197, 197])

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]，这里的reshape操作就是变相的将多头的嵌入维度进行拼接，从而获得长序列向量
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # 这里乘V就是加权求和操作，size[96,197,768](三视角)
        x = self.proj(x)  # 全连接层，size[96,197,768](三视角)
        x = self.proj_drop(x)  # dropout的功能，size[96,197,768](三视角)
        AA = 0  # torch.Size([64, 197, 768])两个视角
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SELU, drop=0.5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 # This 4 represents the dilation multiple of the original number of input nodes after an mlp.
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.5,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.ReLU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, view_nums):
        super(CrossAttention, self).__init__()
        self.view_nums = view_nums
        self.qkv = nn.Linear(64, 128, bias=False)  # 如果后面通道数变了就不是128
        self.attn_drop = nn.Dropout(0.5)
        self.attn_drop = nn.Dropout(0.5)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.5)
        self.dim = dim

    def forward(self, x):
        main_view_features_all = []
        other_views_features_all = []
        B, HW, C = x.shape
        assert B % self.view_nums == 0, f"Input tensor dimension mismatch. Expected HW to be divisible by {self.num_views}"
        for main_view in range(0, B, self.view_nums):
            for j in range(main_view, main_view + self.view_nums):
                # print('main_view：{}'.format(main_view))
                main_view_features = x[main_view, :, :]  # size [784,128]
                # Expand two dimensions into three dimensions size[1,784,128]
                main_view_features = torch.unsqueeze(main_view_features, dim=0)
                main_view_features_all.append(main_view_features)
                # print('j：{}'.format(j))
                # Extract features from other viewpoints, the other viewpoints are stitched together first,
                # and for each additional viewpoint, the image takes on an additional dimension,
                # see the paper for details
                other_views_indices = [i for i in range(main_view, main_view + self.view_nums, 1) if i != j]
                other_views_features = torch.cat([torch.unsqueeze(x[i, :, :], dim=0) for i in other_views_indices],
                                                 dim=0)
                other_views_features = other_views_features.reshape(1, 784, -1)
                other_views_features_all.append(other_views_features)

        main_view_features_all = torch.stack(main_view_features_all)
        other_views_features_all = torch.stack(other_views_features_all)
        q_main = self.qkv(main_view_features_all).transpose(1, 2)  # torch.Size([1, 128, 784])
        k_other = self.qkv(other_views_features_all).transpose(1, 2)  # torch.Size([2, 128, 784])
        v_other = self.qkv(other_views_features_all).transpose(1, 2)  # torch.Size([2, 128, 784])

        # Compute the attention weights between main_view_features and other_views_features
        attn_weights = (q_main @ k_other.transpose(-2, -1))  # torch.Size([96, 8, 147, 294])
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        cross_x = (attn_weights @ v_other).transpose(1, 2).reshape(B, -1, 128)  # torch.Size([96, 147, 128])
        aa = 0
        return cross_x, attn_weights


class multi_processing(nn.Module):
    def __init__(self, num_classes=8, embed_dim=768, distilled=False, attention_layer=CrossAttention, view_nums=3,
                 view_encoder_layer=view_encoder, resnetBlock=BasicBlock):
        super().__init__()
        self.device = 'cuda:0'
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        self.view_encoder_layer = view_encoder_layer()
        self.attention_layer = attention_layer(dim=embed_dim, view_nums=view_nums)
        self.BasicBlock = resnetBlock

    def forward(self, x):
        # --------------------------Steps in drawing a diagram--------------------------------------------------------
        # images, labels_single, labels_mutil_view, view_num = x
        # labels = labels_mutil_view  # 多视角的标签，主要是用来叠加结果的
        # labels_loss = labels_single  # 由于loss经过多视角叠加后，只有单视角的数量，因此计算多视角的时候，这个还是要的
        # pred_all = []
        # feature_view_list = []  # 存储每一个目标的三视角的
        # feature_view_batch = []  # 构成batch的形式
        # for i in range(0, labels.shape[0], view_num):
        # images=[64,3,224,224]，x_images.shape=[3,3,224,224],下面压缩的作用主要是使得图像
        # - `.unsqueeze(1)` 在通道维度上增加一个维度，变成 `[batch_size, 1, channel, height, width]` 的形状。
        # - `.repeat(1, 3, 1, 1)` 将通道维度重复三次，变成 `[batch_size, 3, channel, height, width]` 的形状。
        # x_images = images[i:i + view_num].to(self.device).unsqueeze(1).repeat(1, 3, 1, 1).to(self.device)

        # x_images = images[i:i + view_num].to(self.device)
        # x_feature_name = f'x_feature_{i // view_num}'  # 动态生成变量名
        # exec(f'{x_feature_name} = self.view_encoder_layer(x_images)')  # 使用 exec 生成变量
        # feature_view_list.append(eval(x_feature_name))  # x_feature_name.shape=[3,64,28,28],第一个值是视角数，第二个是通道，第三四是尺寸
        # feature_view = torch.cat(feature_view_list, dim=1)
        # feature_view_list = []  # 重置 feature_view_list,不重置的话，会累次叠加，重置每次就只加3个视角
        # feature_view_batch.append(feature_view)
        # view_encoder_feature = torch.stack(feature_view_batch)
        # x = rearrange(view_encoder_feature, 'b n c h w -> b (n h w) c')
        # origin_vision(x)#
        # # ------------------------------------------------------------------------------------------------------------
        x = self.view_encoder_layer(x)  # x输入[batch,channel,width,height]
        # view_encoder_vision(x)  # Drawing feature maps cnn
        # # -------------------------Cross-attention mechanism----------------------------------------------------------
        x = rearrange(x, 'b c h w -> b (h w) c')  # b=视角*batch，两视角：torch.Size([64, 784, 128])
        x, attn_weights = self.cross_attention_layer(x)
        # atten_feature_vision(x)  # 画特征图自注意力机制
        # attn_weights_vision(attn_weights)  # 画自注意力机制权重图
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        # ------------------------------------------------------------------------
        x = x.permute(0, 2, 1)
        x = x.view(-1, x.shape[1], int(x.shape[2] ** 0.5),
                   int(x.shape[2] ** 0.5))  # 这两句把x变成[B, C, H, W]，两视角：torch.Size([64, 128, 28, 28])
        return x


class VisionTransformer_mutilloss(nn.Module):
    def __init__(self, img_size_w=128, img_size_h=64, patch_size=16, in_c=3, num_classes=8,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, view_num=3,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.0,
                 attn_drop_ratio=0.0, device='cuda:0', drop_path_ratio=0.0, embed_layer=PatchEmbed,
                 pre_layer=multi_processing, attention_layer=CrossAttention, view_encoder_layer=view_encoder,
                 norm_layer=None, act_layer=None):  #
        super(VisionTransformer_mutilloss, self).__init__()
        self.device = 'cuda:0'
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        # self.view_encoder_layer = view_encoder_layer()
        self.multi_layer = pre_layer(num_classes=self.num_classes, embed_dim=768, distilled=False,
                                     attention_layer=attention_layer, view_encoder_layer=view_encoder_layer,
                                     view_nums=view_num)
        self.patch_embed = embed_layer(img_size_w=img_size_w, img_size_h=img_size_h, patch_size=patch_size, in_c=in_c,
                                       embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches  # num_patches=14*14=196

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # Dimension of location coded information to be consistent with patch_embed
        self.pos_embed = nn.Parameter(torch.zeros(1, 196 + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        # Construct an isotropic sequence that contains a total of depth elements. stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        # 创建了一个block的列表，循环次数为depth
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)  # 循环次数为depth
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        x = self.patch_embed(self.multi_layer(x))  # [B, 196, 768], 这个768是经过patch_embed得到的，view_num=1这个量没用到
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # [1,768]
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)  # [197,768]
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def vit_gmvt_patchx_62_128(num_classes: int = 9, view_num: int = 4, has_logits: bool = True):
    model = VisionTransformer_mutilloss(img_size_w=128,
                                        img_size_h=64,
                                        patch_size=16,
                                        in_c=128,
                                        embed_dim=768,
                                        depth=12,
                                        num_heads=12, device='cuda:0',
                                        view_num=view_num,
                                        representation_size=None if has_logits else None,
                                        num_classes=num_classes)
    return model


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
