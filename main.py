from tqdm import tqdm
import utils
import os
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from unet_org import UNetOrg

test_only=False
total_itrs=30000
val_interval=500
save_val_results = True
crop_size = 512
batch_size = 4
val_batch_size = 8

num_classes = 21

def get_dataset():
    """ Dataset And Augmentation
    """
    train_transform = et.ExtCompose([
        et.ExtResize(size=crop_size),
        # et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(crop_size, crop_size), pad_if_needed=False),
        # et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        # et.ExtNormalize(mean=[0.485, 0.456, 0.406],
        #                 std=[0.229, 0.224, 0.225]),
    ])

    val_transform = et.ExtCompose([
        et.ExtResize(size=crop_size),
        et.ExtRandomCrop(size=(crop_size, crop_size), pad_if_needed=False),
        #et.ExtCenterCrop(crop_size),
        et.ExtToTensor(),
        # et.ExtNormalize(mean=[0.485, 0.456, 0.406],
        #                 std=[0.229, 0.224, 0.225]),
    ])
    
    train_dst = VOCSegmentation(image_set='train', transform=train_transform)
    val_dst = VOCSegmentation(image_set='val', transform=val_transform)

    return train_dst, val_dst


def validate(model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

        score = metrics.get_results()
    return score, ret_samples


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    val_batch_size = 1

    train_dst, val_dst = get_dataset()
    train_loader = data.DataLoader(
        train_dst, batch_size=batch_size, shuffle=True, num_workers=2,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: VOC, Train set: %d, Val set: %d" %
          (len(train_dst), len(val_dst)))

    model = UNetOrg(classes=num_classes)

    # Set up metrics
    metrics = StreamSegMetrics(num_classes)

    # Set up optimizer
    # optimizer = torch.optim.SGD(params=[
    #     {'params': model.parameters(), 'lr': 0.1 * opts.lr},
    #     {'params': model.parameters(), 'lr': opts.lr},
    # ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)

    # Set up criterion
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    print("[!] Retrain")
    model = nn.DataParallel(model)
    model.to(device)

    # ==========   Train Loop   ==========#

    if test_only:
        model.eval()
        val_score, ret_samples = validate(
            model=model, loader=val_loader, device=device, metrics=metrics)#, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    interval_loss = 0
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % val_interval == 0:
                save_ckpt('checkpoints/latest_%s_%s.pth' %
                          ("UNet", 'VOC'))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    model=model, loader=val_loader, device=device, metrics=metrics)#,
                    #ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                
                model.train()
            scheduler.step()

            if cur_itrs >= total_itrs:
                return

if __name__ == '__main__':
    main()
