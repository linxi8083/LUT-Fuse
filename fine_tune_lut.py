import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from data.o_fusion_dataset import DistillDataSet
from data.o_fusion_dataset import RandomCropPair
import os
import argparse

import transforms as T
from scripts.loss_lut import fusion_loss
from itertools import chain
from scripts.calculate import OptimizableLUT, Generator_for_info, apply_fusion_4d_with_interpolation

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class TV_4D(nn.Module):
    def __init__(self, dim=16, output_channels=3):
        super(TV_4D, self).__init__()

        self.weight_r = torch.ones(dim, dim, dim, dim - 1, output_channels, dtype=torch.float)
        self.weight_r[..., (0, dim - 2), :] *= 2.0

        self.weight_g = torch.ones(dim, dim, dim - 1, dim, output_channels, dtype=torch.float)
        self.weight_g[..., (0, dim - 2), :, :] *= 2.0

        self.weight_b = torch.ones(dim, dim - 1, dim, dim, output_channels, dtype=torch.float)
        self.weight_b[..., (0, dim - 2), :, :, :] *= 2.0

        self.weight_ir = torch.ones(dim - 1, dim, dim, dim, output_channels, dtype=torch.float)
        self.weight_ir[(0, dim - 2), :, :, :, :] *= 2.0

        self.relu = torch.nn.ReLU()

    def forward(self, LUT):
        device = LUT.device

        self.weight_r = self.weight_r.to(device)

        self.weight_g = self.weight_g.to(device)
        self.weight_b = self.weight_b.to(device)
        self.weight_ir = self.weight_ir.to(device)

        dif_r = LUT[:, :, :, :-1, :] - LUT[:, :, :, 1:, :]
        dif_g = LUT[:, :, :-1, :, :] - LUT[:, :, 1:, :, :]
        dif_b = LUT[:, :-1, :, :, :] - LUT[:, 1:, :, :, :]
        dif_ir = LUT[:-1, :, :, :, :] - LUT[1:, :, :, :, :]

        tv = (torch.mean(torch.mul(dif_r ** 2, self.weight_r)) + torch.mean(torch.mul(dif_g ** 2, self.weight_g)) +
              torch.mean(torch.mul(dif_b ** 2, self.weight_b)) + torch.mean(torch.mul(dif_ir ** 2, self.weight_ir)))

        mn = (torch.mean(self.relu(dif_r)) + torch.mean(self.relu(dif_g)) +
              torch.mean(self.relu(dif_b)) + torch.mean(self.relu(dif_ir)))

        return tv, mn


def fine_tune_lut(lut_model, Generator_context, train_loader, val_loader, device,
                  epochs, learning_rate, experiment_dir, start_epoch=0,
                  optimizer_state=None):
    TV4 = TV_4D().to(device)
    best_val_loss = 1e5
    Generator_context.train()
    loss_fuction = fusion_loss()
    optimizer = optim.Adam(chain(lut_model.parameters(), Generator_context.parameters()), lr=learning_rate)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    # optimizer = optim.Adam(lut_model.parameters(), lr=learning_rate)

    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, epochs):
        lut_model.train()
        Generator_context.train()

        train_loss = 0
        # train_loss_max = 0
        # train_loss_text = 0
        train_loss_l1 = 0
        train_loss_ssim = 0
        train_loss_tv0 = 0
        train_loss_mn0 = 0

        for step, data in enumerate(train_loader):
            I_A, I_B, fuse, _ = data
            optimizer.zero_grad(set_to_none=True)

            if torch.cuda.is_available():
                I_A = I_A.to(device)
                I_B = I_B.to(device)
                high_quality = fuse.to(device)
                loss_fuction = loss_fuction.to(device)

            lut = lut_model()

            tv0, mn0 = TV4(lut)
            loss_tv0 = tv0
            loss_mn0 = mn0

            outputs = apply_fusion_4d_with_interpolation(I_A * 255., I_B * 255., lut, Generator_context)

            l1 = F.l1_loss(outputs, high_quality)
            ssim = loss_fuction(I_A, I_B, outputs)
            loss_all = l1 + ssim + 10.0 * loss_mn0 + 0.0001 * loss_tv0 #+ text_loss + loss_max

            loss_all.backward()
            optimizer.step()

            train_loss += loss_all.item()
            train_loss_l1 += l1.item()
            train_loss_ssim += ssim.item()
            # train_loss_text += text_loss.item()
            # train_loss_max += loss_max.item()
            train_loss_tv0 += loss_tv0.item()
            train_loss_mn0 += loss_mn0.item()
            # train_loss_color += loss_color.item()

        tb_writer.add_scalar("train_total_loss", train_loss/len(train_loader), epoch)
        tb_writer.add_scalar("train_loss_l1", train_loss_l1/len(train_loader), epoch)
        tb_writer.add_scalar("train_loss_ssim", train_loss_ssim / len(train_loader), epoch)
        # tb_writer.add_scalar("train_loss_text", train_loss_text / len(train_loader), epoch)
        # tb_writer.add_scalar("train_loss_max", train_loss_max / len(train_loader), epoch)
        tb_writer.add_scalar("train_loss_tv0", train_loss_tv0/len(train_loader), epoch)
        tb_writer.add_scalar("train_loss_mn0", train_loss_mn0/len(train_loader), epoch)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss / len(train_loader):.6f} - loss_l1: {train_loss_l1 / len(train_loader):.6f} - loss_ssim: {train_loss_ssim / len(train_loader):.6f} - loss_tv: {train_loss_tv0 / len(train_loader):.6f} - loss_tv: {train_loss_tv0 / len(train_loader):.6f} ")
        # print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss / len(train_loader):.6f} - l1: {train_loss_l1 / len(train_loader):.6f} - loss_text: {train_loss_text / len(train_loader):.6f} - loss_max: {train_loss_max / len(train_loader):.6f}")
        # print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss / len(train_loader):.6f} - l1: {train_loss_l1 / len(train_loader):.6f} - tv: {train_loss_tv0 / len(train_loader):.6f} - mn: {train_loss_mn0 / len(train_loader):.6f}")

        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            val_loss, val_loss_l1, val_loss_ssim, val_loss_tv0, val_loss_mn0 = validate_lut(
                lut_model, Generator_context, val_loader, device)
            tb_writer.add_scalar("val_total_loss", val_loss/len(val_loader), epoch)
            tb_writer.add_scalar("val_loss_l1", val_loss_l1/len(val_loader), epoch)
            tb_writer.add_scalar("val_loss_ssim", val_loss_ssim / len(val_loader), epoch)
            # tb_writer.add_scalar("val_loss_text", val_loss_text / len(val_loader), epoch)
            # tb_writer.add_scalar("val_loss_max", val_loss_max / len(val_loader), epoch)
            tb_writer.add_scalar("val_loss_tv0", val_loss_tv0/len(val_loader), epoch)
            tb_writer.add_scalar("val_loss_mn0", val_loss_mn0/len(val_loader), epoch)
            # print(f"Validation - Epoch {epoch} - Loss: {val_loss / len(val_loader):.6f} - l1: {val_loss_l1 / len(val_loader):.6f} - tv: {val_loss_tv0 / len(val_loader):.6f} - mn: {val_loss_mn0 / len(val_loader):.6f}")

            if val_loss < best_val_loss :
                best_val_loss = val_loss
                filename = f"fine_tuned_ygcy_epoch{epoch}_valloss{val_loss:.6f}.npy"
                full_path = os.path.join(checkpoint_dir, filename)
                save_lut(lut_model, full_path)

                context_filename = f"generator_context_epoch{epoch}_valloss{val_loss:.6f}.pth"
                generator_context_save_path = os.path.join(checkpoint_dir, context_filename)
                save_generator_context(Generator_context, save_path=generator_context_save_path)

            print(f"Validation - Epoch {epoch} - Loss: {val_loss / len(val_loader):.6f} - l1: {val_loss_l1 / len(val_loader):.6f} - loss_ssim: {val_loss_ssim / len(val_loader):.6f} - loss_tv0: {val_loss_tv0 / len(val_loader):.6f} - loss_mn0: {val_loss_mn0 / len(val_loader):.6f}")

        checkpoint = {
            "epoch": epoch + 1,
            "lut": lut_model.state_dict(),
            "generator_context": Generator_context.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, "latest.pth"))


def save_lut(lut_module, path):

    lut_weights = lut_module().detach().cpu().numpy()
    np.save(path, lut_weights)
    print(f"Fine-tuned LUT saved to {path}")


def validate_lut(lut_module, generator_context, val_loader, device):
    train_loss = 0
    train_loss_mn0 = 0
    train_loss_tv0 = 0
    train_loss_ssim = 0
    train_loss_l1 = 0
    # train_loss_text = 0
    TV4 = TV_4D().to(device)

    loss_fuction = fusion_loss()
    generator_context.eval()

    lut = lut_module()
    with torch.no_grad():
        for step, data in enumerate(val_loader):
            I_A, I_B, fuse, task = data
            if torch.cuda.is_available():
                I_A = I_A.to(device)
                I_B = I_B.to(device)
                high_quality = fuse.to(device)
                loss_fuction = loss_fuction.to(device)

            outputs = apply_fusion_4d_with_interpolation(I_A * 255., I_B * 255., lut, generator_context)
            tv0, mn0 = TV4(lut)
            loss_tv0 = tv0
            loss_mn0 = mn0
            l1 = F.l1_loss(outputs, high_quality)
            loss_ssim = loss_fuction(I_A, I_B, outputs)
            loss_all = l1 + loss_ssim + 10.0 * loss_mn0 + 0.0001 * loss_tv0

            train_loss += loss_all.item()
            train_loss_l1 += l1.item()
            train_loss_ssim += loss_ssim.item()
            # train_loss_text += text_loss.item()
            # train_loss_max += max_loss.item()
            train_loss_tv0 += loss_tv0.item()
            train_loss_mn0 += loss_mn0.item()

    return train_loss, train_loss_l1, train_loss_ssim, train_loss_tv0, train_loss_mn0


def save_generator_context(generator_context, save_path="generator_context.pth"):
    torch.save(generator_context.state_dict(), save_path)
    print(f"Generator_for_info weights saved to {save_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LUT-Fuse")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", default="experiments")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--lut-init", default="ckpts/fine_tuned_lut_original.npy")
    parser.add_argument("--context-init", default="ckpts/generator_context_original.pth")
    return parser.parse_args()


def ensure_dataset(paths):
    counts = []
    for path in paths:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Dataset directory does not exist: {path}")
        count = len([name for name in os.listdir(path)
                     if os.path.isfile(os.path.join(path, name))])
        if count == 0:
            raise ValueError(f"Dataset directory is empty: {path}")
        counts.append(count)
    if len(set(counts[:3])) != 1:
        raise ValueError(f"Train set counts do not match: {counts[:3]}")
    if len(set(counts[3:])) != 1:
        raise ValueError(f"Validation set counts do not match: {counts[3:]}")


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this training script")

    experiment_dir = os.path.join(args.output_root, args.experiment)
    file_log_path = os.path.join(experiment_dir, "logs")
    os.makedirs(file_log_path, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "results"), exist_ok=True)
    tb_writer = SummaryWriter(log_dir=file_log_path)

    DEVICE = torch.device("cuda")
    train_root = os.path.join(args.data_root, "train")
    test_root = os.path.join(args.data_root, "test")
    visible_path = os.path.join(train_root, "Visible")
    infrared_path = os.path.join(train_root, "Infrared")
    train_fusion_path = os.path.join(train_root, "Fuse_ref")
    test_visible_path = os.path.join(test_root, "Visible")
    test_infrared_path = os.path.join(test_root, "Infrared")
    test_fusion_path = os.path.join(test_root, "Fuse_ref")
    ensure_dataset([visible_path, infrared_path, train_fusion_path,
                    test_visible_path, test_infrared_path, test_fusion_path])

    lut_tensor = torch.tensor(np.load(args.lut_init).astype(np.float32), device=DEVICE)
    lut = OptimizableLUT(lut_tensor)
    Generator_context = Generator_for_info().to(DEVICE)
    Generator_context.load_state_dict(torch.load(args.context_init, map_location=DEVICE))

    start_epoch = 0
    optimizer_state = None
    if args.resume:
        resume = torch.load(args.resume, map_location=DEVICE)
        lut.load_state_dict(resume["lut"])
        Generator_context.load_state_dict(resume["generator_context"])
        optimizer_state = resume["optimizer"]
        start_epoch = resume["epoch"]

    nw = args.num_workers
    data_transform = {
        "train": RandomCropPair(size=(args.crop_size, args.crop_size)),
        "val": T.Compose([T.Resize_16(),
                          T.ToTensor()])}

    train_dataset = DistillDataSet(visible_path=visible_path,
                                  infrared_path=infrared_path,
                                  other_fuse_path=train_fusion_path,
                                  phase="train",
                                  transform=data_transform["train"])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_dataset = DistillDataSet(visible_path=test_visible_path,
                                infrared_path=test_infrared_path,
                                other_fuse_path=test_fusion_path,
                                phase="val",
                                transform=data_transform["val"])
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    fine_tune_lut(lut, Generator_context, train_loader, val_loader, DEVICE,
                  epochs=args.epochs, learning_rate=args.learning_rate,
                  experiment_dir=experiment_dir, start_epoch=start_epoch,
                  optimizer_state=optimizer_state)
