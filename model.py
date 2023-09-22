import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from vision_transformer import vit_base
import math
import random
from torchvision import transforms
from PIL import ImageOps, ImageFilter

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        # x = x.detach()
        logits = self.last_layer(x)
        return nn.functional.normalize(x_proj), logits


class SimMatchV2(nn.Module):
    def __init__(self, num_classes=200, dim=128, K=128*30, args=None):
        super(SimMatchV2, self).__init__()

        self.num_classes = num_classes

        # 设置特征维度
        self.dim = dim
        # 设置主模型
        backbone = vit_base()
        weights_path = "/wang_hp/zhy/gcd-task/pretrained/DINO/dino_vitbase16_pretrain.pth"
        state_dict = torch.load(weights_path)
        backbone.load_state_dict(state_dict)
        for param in backbone.parameters():
            param.requires_grad = False
        for name, param in backbone.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])  
                if block_num >= args.grad_from_block:
                    param.requires_grad = True
                    print(f'Finetuning layer {name}')
        projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers, bottleneck_dim=128)
        self.main = nn.Sequential(backbone, projector)


        # 设置存储在内存中的样本数量
        self.K = K
    
        # 注册一个指针用于指向无标签的内存银行
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

        # 创建一个内存一行用于存储无标签的嵌入
        self.register_buffer("u_bank", torch.randn(self.K, dim))
        # 对无标签的内存银行进行归一化处理
        self.u_bank = nn.functional.normalize(self.u_bank)

        # 创建一个内存银行来存储模型对无标签数据的预测
        self.register_buffer("u_labels", torch.zeros(self.K, num_classes) / num_classes)
    
    @torch.no_grad()
    def _update_unlabel_bank(self, k, prob):
        batch_size = k.size(0)
        ptr = int(self.ptr[0])
        assert self.K % batch_size == 0
        self.u_bank[ptr:ptr + batch_size] = k
        self.u_labels[ptr:ptr + batch_size] = prob
        self.ptr[0] = (ptr + batch_size) % self.K

    # images_w, images_s, label_mask
    def forward(self, images_w, images_s=None):
        # 如果没有提供无标签数据，则只处理有标签数据
        if images_s is None:
            proj, logits = self.main(images_w)
            return proj, logits
        # 获取设备信息
        device = images_w.device

        # 克隆并分离存储在模型中的标签和无标签的内存库
        # l_bank   = self.l_bank.clone().detach()
        # l_labels = self.l_labels.clone().detach()
        u_bank   = self.u_bank.clone().detach()
        u_labels = self.u_labels.clone().detach()

        # 获取有标签和无标签数据的批次大小
        # batch_x = im_x.shape[0]
        # batch_u = images_w.shape[0]
        # print("batch_x.shape:", batch_x.shape)
        # print("batch_u.shape:", batch_x.shape)

        # 获取强数据增强的数量
        num_strong = len(images_s)
        
        with torch.no_grad():
            # 将有标签和弱增强无标签的数据合并
            # im = torch.cat([im_x, im_u_w])
            # 使用EMA模型获取输出
            # if self.eman:
            feat_w, logits_w = self.main(images_w)
            # else:
                # im, idx_unshuffle = self._batch_shuffle_ddp(im)
                # logits_w, feat_w = self.ema(images_w)
                # logits_k = self._batch_unshuffle_ddp(logits_k, idx_unshuffle)
                # feat_k   = self._batch_unshuffle_ddp(feat_k , idx_unshuffle)

            # 分离有标签和无标签的特征
            # feat_kx = feat_k[:batch_x]
            # feat_ku = feat_k[batch_x:]
            # 获取无标签数据的概率输出（伪标签）
            prob_w = F.softmax(logits_w, dim=1)
            # if args.DA:
            #     prob_ku = self.distribution_alignment(prob_ku, args)
        
            # 计算无标签数据和无标签内存库特征相似性
            simmatrix_w = feat_w @ u_bank.T
            relation_w = F.softmax(simmatrix_w / 0.1, dim=-1)
            
        # 将有标签和强增强无标签的数据合并，并获取输出
        feat_w, logits_w = self.main(images_w)
        feat_s, logits_s = self.main(images_s[0])
        # 分离有标签和强增强无标签的输出
        logits_s_list = [logits_s]
        feat_s_list = [feat_s]
        
        for im_q in images_s[1:]:
            feat_s_q, logits_s_q = self.main(im_q)
            logits_s_list.append(logits_s_q)
            feat_s_list.append(feat_s_q)
        
        loss_ee = 0
        loss_ne = 0
        for idx in range(num_strong):
            relation_s_q = F.softmax(feat_s_list[idx] @ u_bank.T / 0.1, dim=1)
            loss_ee += torch.sum(-relation_s_q.log() * relation_w.detach(), dim=1).mean()

            nn_qu = relation_s_q @ u_labels
            loss_ne += torch.sum(-nn_qu.log() * prob_w.detach(), dim=1).mean()
        
        loss_ee /= num_strong
        loss_ne /= num_strong
        
        if torch.isnan(loss_ee) or torch.isinf(loss_ee):
            loss_ee = torch.tensor(0.0, device=device)

        if torch.isnan(loss_ne) or torch.isinf(loss_ne):
            loss_ne = torch.tensor(0.0, device=device)

        # self._update_label_bank(feat_kx, index)
        self._update_unlabel_bank(feat_w, prob_w)

        logits_s_list = torch.cat(logits_s_list)
        logits_s_list = F.softmax(logits_s_list, dim=1)

        prob_w = prob_w.detach().repeat([num_strong, 1])

        loss_nn = torch.sum(-F.log_softmax(logits_s_list, dim=1) * prob_w.detach(), dim=1).mean()
        if torch.isnan(loss_nn) or torch.isinf(loss_nn):
            loss_nn = torch.tensor(0.0, device=device)
        # print("len strong:",num_strong)
        # logits_w = torch.cat((logits_w, logits_s_list), dim=0)
        return feat_w, logits_w, loss_ee, loss_ne, loss_nn



class Solarize(object):
    def __call__(self, img):
        return ImageOps.solarize(img)

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_dino_aug(size=224, scale=(0.2, 1.0), gaussian=0.5, solarize=0.5):
    augs = [
        transforms.RandomResizedCrop(size, scale=scale),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ]

    if gaussian > 0:
        augs.append(transforms.RandomApply([GaussianBlur([.1, 2.])], p=gaussian))
    
    if solarize > 0:
        augs.append(transforms.RandomApply([Solarize()], p=solarize))
    
    augs.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transforms.Compose(augs)

weak_aug = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

moco_aug = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# my_aug = transforms.Compose([
#             transforms.Resize(int(224 / 0.875), 3),
#             transforms.RandomCrop(224),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.ColorJitter(),
#             transforms.RandomGrayscale(p=0.2),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=torch.tensor(mean),
#                 std=torch.tensor(std))
#         ])


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views
        # self.strong_transform = get_dino_aug(size=224, scale=(0.140, 1.000), gaussian=0.5, solarize=0.1)
        interpolation = 3
        crop_pct = 0.875
        image_size = 224
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.strong_transform = transforms.Compose([transforms.Resize(int(image_size / crop_pct), interpolation),
                                                    transforms.RandomCrop(image_size),
                                                    transforms.RandomHorizontalFlip(p=0.5),
                                                    transforms.ColorJitter(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(
                                                        mean=torch.tensor(mean),
                                                        std=torch.tensor(std))
                                                ])

    def __call__(self, x):
        # if not isinstance(self.base_transform, list):
        #     return [self.base_transform(x), self.base_transform(x), moco_aug(x)]
        # else:
        #     return [self.base_transform(x), self.base_transform(x), moco_aug(x)]
        if not isinstance(self.base_transform, list):
            return [self.base_transform(x) for i in range(self.n_views)]
        else:
            return [self.base_transform[i](x) for i in range(self.n_views)]

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



def info_nce_logits(features, n_views=2, temperature=1.0, device='cuda'):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

def teacher_distribution(teacher_logits: torch.Tensor) -> torch.Tensor:
    """计算目标分布

    Args:
        teacher_logits: Input teacher logits.

    Returns:
        torch.Tensor: teacher distribution.
    """
    weight = teacher_logits**2 / teacher_logits.sum(0)
    return (weight.t() / weight.sum(1)).t()  

# linea-rise-fall-1
# def compute_value(epoch):
#     if 0 <= epoch <= 10:
#         return 1 + 0.05 * epoch
#     elif 10 < epoch <= 50:
#         return 1.5 - 0.0125 * (epoch - 10)
#     elif epoch > 50:
#         return 1
#     else:
#         raise ValueError("Epoch value out of range!")

# linear-rise-fall-2
# def compute_value(epoch):
#     if 0 <= epoch <= 20:
#         return 1 + 0.015 * epoch
#     elif 20 < epoch <= 50:
#         return 1.3 - 0.01 * (epoch - 20)
#     elif epoch > 50:
#         return 1
#     else:
#         raise ValueError("Epoch value out of range!")
# rise-rise
# def compute_value(epoch):
#     if 0 <= epoch <= 5:
#         return 1 + 0.02 * epoch
#     elif 5 < epoch <= 50:
#         return 1+0.005 * (epoch-5)
#     elif epoch > 50:
#         return 1+0.005 * (50-5)
#     else:
#         raise ValueError("Epoch value out of range!")
# rise-rise-2

# def scale_factor(epoch):
#     if epoch < 10:
#         return (0.5)**epoch + 1
#     else:
#         base_value = (0.5)**10 + 1  # 计算epoch=10时的值
#         return base_value+(-0.1*math.log2(10))+0.1*math.log2(epoch)  # 从epoch=10开始应用对数函数

# def scale_factor(epoch):
#     return math.floor((0.7)**epoch * 40 + 10)
# def scale_factor(epoch):
#     return math.floor((0.7)**epoch * 49 + 1)

# def replace_values_with_epoch(teacher_output, epoch):
#     device = teacher_output.device  # Get the device of teacher_output
#     # if epoch < 10:
#     #     num_values_to_replace = 0
#     # elif 10 <= epoch < 50:  # 40 epochs to increase from 1 to 20
#     #     num_values_to_replace = 1 + (epoch - 10) * 19 // 39  # Linearly increase from 1 to 20 over 40 epochs
#     # else:
#     #     num_values_to_replace = 20  # Keep replacing 20 values for epochs >= 50
#     if epoch < 50:
#         num_values_to_replace = 1 + (epoch * 19) // 49  # Linearly increase from 1 to 20 over 50 epochs
#     else:
#         num_values_to_replace = 20
#     # if epoch == 0:
#     #     num_values_to_replace = 0
#     # elif 0< epoch < 50:
#     #     num_values_to_replace = 1 + (epoch * 19) // 49  # Linearly increase from 1 to 20 over 70 epochs
#     # else:
#     #     num_values_to_replace = 20
#     # epoch = epoch - 25
#     # # Determine the number of values to replace based on the epoch
#     # if epoch < 0:
#     #     num_values_to_replace = 0
#     # elif 0<= epoch <=75:
#     #     num_values_to_replace = 1 + (epoch * 19) // 49  # Linearly increase from 1 to 20 over 50 epochs
#     # else: 
#     #     num_values_to_replace = 20  # Keep replacing 20 values for epochs >= 50
#     # num_values_to_replace = scale_factor(epoch)

#     # Sort the values in each row and get the indices of the smallest values
#     _, indices = torch.sort(teacher_output, dim=1)
#     smallest_indices = indices[:, :num_values_to_replace]

#     # Create a tensor with -e^9 values and ensure it's on the same device as teacher_output
#     replacement_values = -torch.exp(torch.tensor(9.0, device=device)).expand_as(smallest_indices)

#     # Use scatter_ to replace the smallest values with -e^9
#     teacher_output.scatter_(1, smallest_indices, replacement_values)

#     return teacher_output

# def replace_values_with_epoch1(epoch):
#     if epoch == 0:
#         num_values_to_replace = 0
#     elif 1 <= epoch < 50:  # 49 epochs to increase from 1 to 10
#         num_values_to_replace = 1 + (epoch - 1) * 9 // 48  # Linearly increase from 1 to 10 over 49 epochs
#     else:
#         num_values_to_replace = 10  # Keep replacing 10 values for epochs >= 50
#     return num_values_to_replace

# def replace_values_with_epoch2(epoch):
#     if epoch == 0:
#         num_values_to_replace = 0
#     elif 1 <= epoch < 50:  # 49 epochs to increase from 1 to 20
#         num_values_to_replace = 1 + (epoch - 1) * 19 // 48  # Linearly increase from 1 to 20 over 49 epochs
#     else:
#         num_values_to_replace = 20  # Keep replacing 20 values for epochs >= 50
#     return num_values_to_replace
def scale_factor1(epoch):
    # return math.floor((0.7)**epoch * 40 + 10)
    if epoch < 50:
        num_values_to_replace = 1 + (epoch * 19) // 49  # Linearly increase from 1 to 20 over 50 epochs
    else:
        num_values_to_replace = 20
    return num_values_to_replace
def scale_factor2(epoch):
    # if epoch < 5:
    #     return math.floor((0.9)**epoch * 30 + 20)
    # else:
    #     return 0
    # return math.floor((0.7)**epoch * 30 + 20)
    if epoch < 50:
        num_values_to_replace = 1 + (epoch * 9) // 49  # Linearly increase from 1 to 20 over 50 epochs
    else:
        num_values_to_replace = 5
    return num_values_to_replace   
# def original_factor(epoch):
#     if epoch < 50:
#         num_values_to_replace = 1 + (epoch * 19) // 49  # Linearly increase from 1 to 20 over 50 epochs
#     else:
#         num_values_to_replace = 20
#     return num_values_to_replace
def replace_values_with_epoch(teacher_output, epoch):
    device = teacher_output.device  # Get the device of teacher_output

    # Get the indices of the maximum values for each row
    _, max_indices = torch.max(teacher_output, dim=1)

    # For each row, determine the number of values to replace and replace them
    for i, max_index in enumerate(max_indices):
        if max_index < 100:
            num_values_to_replace = scale_factor1(epoch)
            _, sorted_indices = torch.sort(teacher_output[i])
            smallest_indices = sorted_indices[:num_values_to_replace]
            teacher_output[i, smallest_indices] = -torch.exp(torch.tensor(9.0, device=device))
        else:
            num_values_to_replace = scale_factor2(epoch)
            # _, sorted_indices = torch.sort(teacher_output[i])
            # smallest_indices = sorted_indices[:num_values_to_replace]
            # teacher_output[i, smallest_indices] = teacher_output[i, smallest_indices]/2
            _, sorted_indices = torch.sort(teacher_output[i])
            smallest_indices = sorted_indices[:num_values_to_replace]
            teacher_output[i, smallest_indices] = -torch.exp(torch.tensor(9.0, device=device))
        # Get the indices of the smallest values for the current row


        # Replace the smallest values with -e^9
        # 


    return teacher_output

def scale_factor(epoch):
    return 1 + 1 / (1 + math.exp(-(epoch-50)*0.1)) /2

class DistillLoss(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs, 
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
    # def forward(self, student_output, epoch):
    #     """
    #     Cross-entropy between softmax outputs of the teacher and student networks.
    #     """
    #     teacher_output = student_output.detach()
    #     student_out = student_output / self.student_temp
    #     # student_out = student_out.chunk(self.ncrops)

    #     # teacher centering and sharpening
    #     temp = self.teacher_temp_schedule[epoch]
    #     # teacher_out = F.softmax(teacher_output / temp, dim=-1)
    #     # 假设teacher_output是一个N*F的二维tensor
    #     _, top5_indices = teacher_output.topk(5, dim=-1)

    #     # 为前五名的值加权
    #     weighting_factor = 2  # 你可以调整这个因子的值
    #     enhanced_output = teacher_output.clone()
    #     for i in range(teacher_output.size(0)):
    #         enhanced_output[i, top5_indices[i]] += weighting_factor

    #     # 使用softmax
    #     teacher_out = F.softmax(enhanced_output / temp, dim=-1)
    #     # teacher_out = teacher_out.detach().chunk(2)

    #     loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)
    #     total_loss = loss.mean()
    #     return total_loss
    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(2)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        

        # max_probs, _ = torch.max(teacher_out, dim=-1)
        # # print(max_probs)
        # mask = max_probs.ge(0.5).float()
        # mask = mask.chunk(2)
        # 获取teacher_out中每行的最大值及其位置索引
        # max_values, max_indices = torch.max(teacher_out, dim=-1)
        # # 根据max_indices的位置创建一个阈值张量
        # thresholds = torch.where(max_indices < 100, torch.tensor(0.7, device=max_indices.device), torch.tensor(0.35, device=max_indices.device))
        # thresholds = thresholds.to(teacher_out.device)  # 确保阈值张量在与teacher_out相同的设备上
        # # 使用teacher_out的最大值和相应的阈值进行比较
        # mask = max_values.ge(thresholds).float()
        # mask = mask.chunk(2)

        teacher_out = teacher_out.chunk(2)
        #***************************************************top5*****************************************#
        # # 假设teacher_output是一个N*F的二维tensor
        # _, top5_indices = teacher_output.topk(5, dim=-1)

        # # 为前五名的值加权
        # weighting_factor = 1  # 你可以调整这个因子的值
        # enhanced_output = teacher_output.clone()
        # for i in range(teacher_output.size(0)):
        #     enhanced_output[i, top5_indices[i]] += weighting_factor

        # # 使用softmax
        # teacher_out = F.softmax(enhanced_output / temp, dim=-1)
        #**************************************************top5*********************************************#
        #*************************************************linear—big-small**********************************#
        # _size = teacher_output.size(0)
        # weights = torch.cat([torch.ones(100), torch.ones(100) * scale_factor(epoch)], dim=0)
        # weights = weights.unsqueeze(0).expand(_size, -1).to(teacher_output.device)
        # # 对teacher_logits进行加权
        # weighted_logits = teacher_output * weights
        # teacher_out = F.softmax(weighted_logits / temp, dim=-1)
        # teacher_out = teacher_out.detach().chunk(2)
        #*************************************************linear—big-small**********************************#
        # 去除最小值
        # modified_output = replace_values_with_epoch(teacher_output, epoch)
        # min_values, min_indices = torch.min(teacher_output, dim=1, keepdim=True)
        # teacher_output.scatter_(1, min_indices, -torch.exp(torch.tensor(9.0,device=teacher_output.device)).expand_as(min_indices))
        # teacher_out = F.softmax(modified_output / temp, dim=-1)
        # teacher_out = teacher_out.chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                #-----------------------original-twoview-co-------------------------
                if v == iq :
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
                #---------------------------------------------------------------------
                # if v != iq:
                #     # we skip cases where student and teacher operate on the same view
                #     # continue
                #     loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                #     total_loss += loss.mean()
                #     n_loss_terms += 1
                #----------------------singleview-self-----------------------------------
                # if v == iq:
                #     loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                #     total_loss += loss.mean()
                #     n_loss_terms += 1
                #----------------------threshold---------------------------------------------------#
                # if v == iq :
                #     # we skip cases where student and teacher operate on the same view
                #     continue
                # loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1) * mask[iq].detach()
                # total_loss += loss.mean()
                # n_loss_terms += 1
        # for v in range(len(student_out)):
        #     loss = torch.sum(-teacher_out[0] * F.log_softmax(student_out[v], dim=-1), dim=-1)
        #     total_loss += loss.mean()
        #     n_loss_terms += 1

        total_loss /= n_loss_terms
        return total_loss

    # def forward(self, student_output, epoch):
    #     """
    #     Cross-entropy between softmax outputs of the teacher and student networks.
    #     """
    #     teacher_out = student_output.detach()
    #     temp = self.teacher_temp_schedule[epoch]
    #     teacher_out = teacher_out / temp
    #     teacher_out = teacher_out / teacher_out.sum(0)
    #     teacher_out = F.softmax(teacher_out, dim=-1)
    #     teacher_out = teacher_out.chunk(2)

    #     student_out = student_output / self.student_temp
    #     # teacher_out = teacher_distribution(student_out.detach())
    #     student_out = student_out.chunk(self.ncrops)

    #     # teacher centering and sharpening
    #     # temp = self.teacher_temp_schedule[epoch]
    #     # teacher_out = F.softmax(teacher_out, dim=-1)


    #     total_loss = 0
    #     n_loss_terms = 0
    #     for iq, q in enumerate(teacher_out):
    #         for v in range(len(student_out)):
    #             if v == iq & iq == 0:
    #                 # we skip cases where student and teacher operate on the same view
    #                 # continue
    #                 loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
    #                 # loss = F.kl_div(student_out[v], q)
    #                 total_loss += loss.mean()
    #                 n_loss_terms += 1
    #     total_loss /= n_loss_terms
    #     return total_loss
    
# ------------------------------------Mine Loss------------------------------------ #
def unsupervised_contrastive_loss(features: torch.Tensor,  
                                  device='cuda') -> torch.Tensor:
    """
    Compute the unsupervised contrastive loss.
    
    Args:
        features (torch.Tensor): Input features tensor, two views of samples.
        device (str): Device to move tensors to.
        
    Returns:
        torch.Tensor: The unsupervised contrastive loss value.
    """
    
    # Pre-compute batch_size
    batch_size = features.size(0) // 2
    total_batch_size = 2 * batch_size
    
    features = F.normalize(features, dim=1)
    # compute similarity matrix
    similarity_matrix = torch.matmul(features, features.T)
    
    # Compute masks for positive and negative pairs
    positive_mask = torch.zeros_like(similarity_matrix, dtype=torch.bool)
    positive_mask[torch.arange(batch_size), torch.arange(batch_size, total_batch_size)] = True
    
    negative_mask = torch.ones_like(similarity_matrix, dtype=torch.bool)
    negative_mask[batch_size:, :] = 0
    negative_mask[:, batch_size:] = 0
    negative_mask[torch.eye(total_batch_size, dtype=torch.bool)] = 0
    
    # Extract positive and negative pairs
    positives = similarity_matrix[positive_mask].view(-1, 1)
    negatives = similarity_matrix[negative_mask].view(batch_size, -1)
    
    # Concatenate and scale logits
    logits = torch.cat([positives, negatives], dim=1)
    
    # Compute the loss
    pseudo_labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    loss = nn.CrossEntropyLoss()(logits, pseudo_labels)
    
    return loss


def supervised_contrastive_loss(features: torch.Tensor, 
                                labels: torch.Tensor, 
                                temperature=0.07, 
                                base_temperature=0.07, 
                                device='cpu') -> torch.Tensor:
    """
    Compute the supervised contrastive loss.
    
    Args:
        features (torch.Tensor): features of shape [batch_size, feature_dim],original view of samples.
        labels (torch.Tensor): labels associated with features
        temperature (float): scaling temperature for contrastive loss
        base_temperature (float): base temperature for scaling
        
    Returns:
        torch.Tensor: computed loss
    """
    epsilon = 1e-7
    
    # Input validation
    if len(features.shape) != 2:
        raise ValueError('`features` needs to be [batch_size, feature_size]')
    if labels is None:
        raise ValueError('Please provide labels for supervised contrastive loss')
    
    batch_size = features.shape[0]
    labels = labels.contiguous().view(-1, 1)
    
    if labels.shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')
    
    # Create the mask
    mask = torch.eq(labels, labels.T).float().to(device)
    
    # Compute similarity_matrix
    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    # Center the logits to prevent overflow
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()
    
    # Construct mask
    eye_mask = torch.eye(batch_size, dtype=torch.bool).to(device)
    logits_mask = ~eye_mask
    mask = mask * logits_mask.float()
    
    # Compute log probabilities
    exp_logits = torch.exp(logits) * logits_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + epsilon)
    
    # Compute mean log probability for positives
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + epsilon)
    
    # Final loss computation
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    return loss.mean()


# 对比聚类
class Network(nn.Module):
    def __init__(self, backbone, feature_dim, class_num):
        super(Network, self).__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.instance_projector = nn.Sequential(
            nn.Linear(768, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(768, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j):
        h_i = self.backbone(x_i)
        h_j = self.backbone(x_j)

        z_i = F.normalize(self.instance_projector(h_i), dim=1)
        z_j = F.normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        h = self.backbone(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c
    
class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss

class MyModel(nn.Module):
    def __init__(self, num_classes=200, feature_dim=256, K=128):
        super(MyModel, self).__init__()
        self.K = K
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        self.register_buffer("u_bank", torch.zeros(self.K, feature_dim))
        # self.u_bank = nn.functional.normalize(self.u_bank)
        self.register_buffer("u_labels", torch.zeros(self.K, num_classes) / num_classes)
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def update_unlabel_bank(self, feature, prob):
        batch_size = feature.size(0)
        ptr = int(self.ptr[0])
        assert self.K % batch_size == 0
        self.u_bank[ptr:ptr + batch_size] = feature
        self.u_labels[ptr:ptr + batch_size] = prob
        self.ptr[0] = (ptr + batch_size) % self.K