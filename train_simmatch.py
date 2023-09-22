import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse

import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model import SimMatchV2, DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups, unsupervised_contrastive_loss, supervised_contrastive_loss
from torchvision import transforms
from PIL import ImageOps, ImageFilter
import random
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
from copy import deepcopy
from data.data_utils import MergedDataset

def scale_factor(epoch):
    return (0.5)**epoch +1

def train(student, train_loader, test_loader, unlabelled_train_loader, args):
    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )


    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )

    # # inductive
    # best_test_acc_lab = 0
    # # transductive
    # best_train_acc_lab = 0
    # best_train_acc_ubl = 0 
    # best_train_acc_all = 0

    for epoch in range(args.epochs):
        loss_record = AverageMeter()

        student.train()
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            # images = torch.cat(images, dim=0).cuda(non_blocking=True)
            # 分离无标签数据的弱增强和强增强图像
            images_w, images_s = images[:1], images[1:]
            images_w = torch.cat(images_w, dim=0).cuda(non_blocking=True)
            # images_w = images_w.cuda(non_blocking=True)
            # print("images.shape:",images.shape)
            for img_idx in range(len(images_s)):
                images_s[img_idx] = images_s[img_idx].cuda(non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                student_proj, student_out, loss_ee, loss_ne, loss_nn= student(images_w, images_s)
                # student_proj, student_out = student(images)
                # teacher_out = student_out.detach()

                # clustering, sup
                # sup_logits = torch.cat([f[mask_lab] for f in (student_out ).chunk(6)], dim=0)
                # sup_labels = torch.cat([class_labels[mask_lab] for _ in range(6)], dim=0)
                # cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)
                cls_loss = nn.CrossEntropyLoss()(student_out[mask_lab], class_labels[mask_lab])

                # clustering, unsup
                cluster_loss = cluster_criterion(student_out.chunk(6)[0], epoch)
                # cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += args.memax_weight * me_max_loss
                # else:

                # _size = teacher_out.size(0)
                # weights = torch.cat([torch.ones(100), torch.ones(100) * scale_factor(epoch+1)], dim=0)
                # weights = weights.unsqueeze(0).expand(_size, -1).to(args.device)
                # # 对teacher_logits进行加权
                # weighted_logits = teacher_out * weights
                # cluster_loss = cluster_criterion(student_out, weighted_logits, epoch)
                # avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                # me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                # cluster_loss += args.memax_weight * me_max_loss

                # # represent learning, unsup
                # contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                # contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # # representation learning, sup
                # student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                # student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                # sup_con_labels = class_labels[mask_lab]
                # sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

                #-----------------------------------Mine Representation Learning Loss----------------------------------#
                # # Calculate self-supervised contrastive loss for all the samples
                # contrastive_loss = unsupervised_contrastive_loss(features=student_proj, device=args.device)

                # # Calculate supervised contrastive loss for the labeled samples
                # student_proj = student_proj.chunk(2)[0][mask_lab]
                # student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                # sup_con_labels = class_labels[mask_lab]
                # sup_con_loss = supervised_contrastive_loss(student_proj, sup_con_labels, device=args.device)

                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                # pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                # pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '
                pstr += f'loss_ee: {loss_ee.item():.4f} '
                pstr += f'loss_ne: {loss_ne.item():.4f} '
                pstr += f'loss_nn: {loss_nn.item():.4f} '
                # print("cls_loss:",cls_loss)
                # print("cluster_loss:",cluster_loss)
                # print("loss_ee:",loss_ee)
                # print("loss_ne:",loss_ne)
                # print("loss_nn:",loss_nn)
                loss = 0
                loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss * 10
                # loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss
                loss += args.sup_weight * loss_ee + args.sup_weight * loss_ne + args.sup_weight *loss_nn
                
            # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        args.logger.info('Testing on unlabelled examples in the training data...')
        all_acc, old_acc, new_acc = test(student, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
        # args.logger.info('Testing on disjoint test set...')
        # all_acc_test, old_acc_test, new_acc_test = test(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)


        args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
        # args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))

        # Step schedule
        exp_lr_scheduler.step()

        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        # torch.save(save_dict, args.model_path)
        # args.logger.info("model saved to {}.".format(args.model_path))

        # if old_acc_test > best_test_acc_lab:
        #     
        #     args.logger.info(f'Best ACC on old Classes on disjoint test set: {old_acc_test:.4f}...')
        #     args.logger.info('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
        #     
        #     torch.save(save_dict, args.model_path[:-3] + f'_best.pt')
        #     args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))
        #     
        #     # inductive
        #     best_test_acc_lab = old_acc_test
        #     # transductive            
        #     best_train_acc_lab = old_acc
        #     best_train_acc_ubl = new_acc
        #     best_train_acc_all = all_acc
        # 
        # args.logger.info(f'Exp Name: {args.exp_name}')
        # args.logger.info(f'Metrics with best model on test set: All: {best_train_acc_all:.4f} Old: {best_train_acc_lab:.4f} New: {best_train_acc_ubl:.4f}')


def test(model, test_loader, epoch, save_name, args):

    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc



# def test(model, test_loader, epoch, save_name, args):
#     model.eval()

#     preds, targets = [], []
#     mask = np.array([])

#     # 打开log.txt文件以追加模式
#     with open('/wang_hp/zhy/SimGCD-main/test/novel_sharpen_log1.txt', 'a') as f:
#         # 写入当前epoch的分隔行
#         f.write(f"****epoch {epoch}*******\n")
#         # 写入标题行
#         f.write("True Label, Predicted Label, Prediction Value, Max Old Class Value, Max New Class Value, Ratio (Old/New)\n")

#         for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
#             images = images.cuda(non_blocking=True)
#             with torch.no_grad():
#                 _, logits = model(images)

#                 # 在此处写入每个样本的信息
#                 for i in range(logits.size(0)):
#                     true_label = label[i].item()
#                     predicted_label = logits[i].argmax().item()
#                     prediction_value = logits[i][predicted_label].item()

#                     # 获取旧类和新类的最大预测值
#                     max_old_class_value = logits[i][:100].max().item()
#                     max_new_class_value = logits[i][100:].max().item()

#                     # 计算旧类和新类的最大预测值的比值
#                     ratio = max_old_class_value / max_new_class_value

#                     # 写入每个样本的信息
#                     f.write(f"{true_label}, {predicted_label}, {prediction_value:.4f}, {max_old_class_value:.4f}, {max_new_class_value:.4f}, {ratio:.4f}\n")

#                 preds.append(logits.argmax(1).cpu().numpy())
#                 targets.append(label.cpu().numpy())
#                 mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

#     preds = np.concatenate(preds)
#     targets = np.concatenate(targets)

#     all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
#                                                     T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
#                                                     args=args)

#     return all_acc, old_acc, new_acc

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

class MultiTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        imgs = []
        for t in self.transforms:
            imgs.append(t(x))
        return imgs

class CustomCub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root='/wang_hp/zhy/gcd-task/data', train=True, transform=None, target_transform=None, loader=default_loader, download=True):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.loader = loader
        self.train = train


        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.uq_idxs = np.array(range(len(self)))

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target, self.uq_idxs[idx]
        return img, target, idx

def subsample_classes(dataset, include_classes=range(160)):

    include_classes_cub = np.array(include_classes) + 1     # CUB classes are indexed 1 --> 200 instead of 0 --> 199
    cls_idxs = [x for x, (_, r) in enumerate(dataset.data.iterrows()) if int(r['target']) in include_classes_cub]

    # TODO: For now have no target transform
    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset

def subsample_instances(dataset, prop_indices_to_subsample=0.5):

    np.random.seed(0)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))

    return subsample_indices

def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.data = dataset.data[mask]
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2b'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='cub', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=False)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=1, type=int)
    parser.add_argument('--exp_name', default='simmatch_first', type=str)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)
    args.device = device

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['simgcd'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    # backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

    # if args.warmup_model_dir is not None:
    #     args.logger.info(f'Loading weights from {args.warmup_model_dir}')
    #     backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))
    
    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    # for m in backbone.parameters():
    #     m.requires_grad = False

    # # Only finetune layers from block 'args.grad_from_block' onwards
    # for name, m in backbone.named_parameters():
    #     if 'block' in name:
    #         block_num = int(name.split('.')[1])
    #         if block_num >= args.grad_from_block:
    #             m.requires_grad = True

    
    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    # train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    # train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    # --------------------
    # DATASETS
    # --------------------
    # train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
    #                                                                                      train_transform,
    #                                                                                      test_transform,
    #  
    weak_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])



    transform_u_list = [
        weak_transform,
        # weak_transform,
        get_dino_aug(size=224, scale=(0.140, 1.000), gaussian=0.5, solarize=0.1),
        get_dino_aug(size=192, scale=(0.117, 0.860), gaussian=0.5, solarize=0.1),
        get_dino_aug(size=160, scale=(0.095, 0.715), gaussian=0.5, solarize=0.1),
        get_dino_aug(size=120, scale=(0.073, 0.571), gaussian=0.5, solarize=0.1),
        get_dino_aug(size=96 , scale=(0.050, 0.429), gaussian=0.5, solarize=0.1),
    ]

    transform_u = MultiTransform(transform_u_list)
    transform_val = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
    whole_training_set = CustomCub2011(transform=transform_u, train=True)

    # 有标签训练集
    # train_dataset_labelled = deepcopy(whole_training_set)
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=range(100))
    subsample_indices = subsample_instances(train_dataset_labelled)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)
    print("len of labeled:", len(train_dataset_labelled))

    # 无标签训练集
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))
    print("len of unlabeled:", len(train_dataset_unlabelled))

    # 验证集
    val_dataset = deepcopy(train_dataset_unlabelled)
    val_dataset.transform = transform_val
    print("len of val:", len(val_dataset))
    # logger.info(val_dataset.transform)

    train_dataset = MergedDataset(train_dataset_labelled, train_dataset_unlabelled)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(val_dataset, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    # test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
    #                                   batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    # projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    # model = nn.Sequential(backbone, projector).to(device)
    model = SimMatchV2(args=args).to(device)
    # ----------------------
    # TRAIN
    # ----------------------
    # train(model, train_loader, test_loader_labelled, test_loader_unlabelled, args)
    train(model, train_loader, None, test_loader_unlabelled, args)
