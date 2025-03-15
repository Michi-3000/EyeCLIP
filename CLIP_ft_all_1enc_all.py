#BATCH_SIZE must larger than 1
import torch
import eyeclip
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import random
import argparse
from tqdm import tqdm
import logging
import time
from datetime import datetime
import pandas as pd
import util.lr_sched as lr_sched
import matplotlib.pyplot as plt
import re

def plot_training_losses(log_file_path, output_image_path):
    epochs = []
    iterations = []
    losses = []
    mae_losses = []
    clip_losses_img = []
    clip_losses_text = []

    log_pattern = re.compile(
        r'Train: Epoch (\d+) - iter (\d+) - loss ([\d.]+) - MAEloss ([\d.]+) - CLIPloss_img ([\d.]+)- CLIPloss_text ([\d.]+)'
    )

    with open(log_file_path, 'r') as log_file:
        for line in log_file:
            match = log_pattern.search(line)
            if match:
                epoch = int(match.group(1))
                iteration = int(match.group(2))
                loss = float(match.group(3))
                mae_loss = float(match.group(4))
                clip_loss_img = float(match.group(5))
                clip_loss_text = float(match.group(6))

                epochs.append(epoch)
                iterations.append(iteration)
                losses.append(loss)
                mae_losses.append(mae_loss)
                clip_losses_img.append(clip_loss_img)
                clip_losses_text.append(clip_loss_text)

    x_labels = [f"{epoch}-{iteration}" for epoch, iteration in zip(epochs, iterations)]

    plt.figure(figsize=(12, 6))
    plt.plot(x_labels, losses, label='Loss')
    plt.plot(x_labels, mae_losses, label='MAEloss')
    plt.plot(x_labels, clip_losses_img, label='CLIPloss_img')
    plt.plot(x_labels, clip_losses_text, label='CLIPloss_text')

    plt.xlabel('Epoch-Iteration')
    plt.ylabel('Loss')
    plt.title('Training Losses Over Epochs and Iterations')
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)

    plt.savefig(output_image_path)



def splitdf(df,col='orgid',test_size=.2):
    if col==None:
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(df, test_size=test_size, random_state=42,stratify=None)
    else:
        from sklearn.model_selection import GroupShuffleSplit
        train_inds, test_inds = next(GroupShuffleSplit(test_size=test_size, n_splits=2, random_state = 7).split(df, groups=df[col]))
        train = df.iloc[train_inds].reset_index(drop=True)
        test = df.iloc[test_inds].reset_index(drop=True)
    return train,test


#ensure when set==val, num(n_rep)!=0


def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="clip", choices=("clip"))

    parser.add_argument(
        "--dataset_path", type=str, default="./data/"
    )
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default="./ft_checkpoints")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--eval", dest="eval", action="store_true")

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=2, metavar='N',
                        help='epochs to warmup LR')

    args = parser.parse_args()

    set_random_seeds(args.seed)
    return args


'''
# preprocess is defined as:
def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),#
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
'''

class CustomDataLoader:
    def __init__(self, datasets, batch_size, shuffle=True):
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.iterators = [iter(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)) for dataset in datasets]

    def __iter__(self):
        self.iterators = [iter(DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=True)) for dataset in self.datasets]
        return self
    
    def __len__(self):
        return len(self.datasets[0])

    def __next__(self):
        results = []
        for it in self.iterators:
            try:
                data = next(it)
                results.append(data)
            except StopIteration:
                results.append(None)
        if all(result is None for result in results):
            raise StopIteration
        return tuple(results)

class MAE_dataset(Dataset):
    def __init__(self, dt, col='orgid'):
        self.dt = dt
        self.orgid = dt.drop_duplicates(subset=[col])[col].to_list()
        self.col = col
    def __len__(self):
        return len(self.orgid)
    def __getitem__(self, idx):
        try:
            sdt = self.dt[self.dt[self.col]==self.orgid[idx]]
            image = preprocess(Image.open(np.random.choice(sdt['impath']))) # Image from PIL module
        except Exception as e:
            print(f'Skipped sample (index {idx}, file {self.orgid[idx]}). {str(e)}')
            return self.__getitem__((idx + 1) % len(self.orgid))
        return image


class img2img_dataset(Dataset):
    def __init__(self, dt, col='orgid'):
        self.dt = dt[dt['n_mod']>1]
        print("filtered dt img2img:", len(self.dt))
        self.orgid = self.dt.drop_duplicates(subset=[col])[col].to_list()
        print("filtered orgid img2img:", len(self.orgid))
        self.col = col

    def __len__(self):
        return len(self.orgid)

    def __getitem__(self, idx):
        try:
            sdt = self.dt[self.dt[self.col]==self.orgid[idx]]
            imgs = np.random.choice(sdt['impath'], size=2, replace=False)
            image1 = preprocess(Image.open(imgs[0]))
            image2 = preprocess(Image.open(imgs[1]))

        except Exception as e:
            print(f'Skipped sample (index {idx}, file {self.orgid[idx]}). {str(e)}')
            return self.__getitem__((idx + 1) % len(self.orgid))
        return image1,image2

class img2text_dataset(Dataset):
    def __init__(self, dt, col='orgid'):
        self.dt = dt[dt['n_rep']!=0]
        print("filtered dt img2text:", len(self.dt))
        self.orgid = self.dt.drop_duplicates(subset=[col])[col].to_list()
        self.col = col

    def __len__(self):
        return len(self.orgid)

    def __getitem__(self, idx):
        try:
            sdt = self.dt[self.dt[self.col]==self.orgid[idx]]
            image = preprocess(Image.open(np.random.choice(sdt['impath'])))
            if not isinstance(sdt['kwe'].to_list()[0], str):
                titles = sdt['ANS'].to_list()[0]
            else:
                titles = sdt['kwe'].to_list()[0]
                
        except Exception as e:
            print(f'Skipped sample (index {idx}, file {self.orgid[idx]}, text {titles}). {str(e)}')
            return self.__getitem__((idx + 1) % len(self.orgid[idx]))
        return image,eyeclip.tokenize(titles, truncate=True)[0]



if __name__ == "__main__":
    args = parse_argument()
    now = datetime.now()

    #model
    device = "cuda" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    print(device)
    
    model, preprocess = eyeclip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training

    #checkpoint = torch.load('CLIP_ft_all_key/epoch1_2000.pt')
    #model.load_state_dict(checkpoint['model_state_dict'])

    print("Successfully load the model!")
    print(preprocess)
    
    suffix = "CLIP_ft_all_key_"+now.strftime("%m-%d-%H%M")
    args.out_dir = os.path.join(args.out_dir, suffix)
    if args.eval and args.checkpoint:
        args.out_dir = '/'.join(args.checkpoint.split('/')[:-1])+"/"+args.checkpoint.split('/')[-1][:-3]+"_test"
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * args.batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / args.batch_size))
    print("actual lr: %.2e" % args.lr)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    with open(os.path.join(args.out_dir, 'args.txt'), 'w') as file:
        for key, value in vars(args).items():
            file.write(f'{key}: {value}\n')
    dt = pd.read_csv("data/clip_mae_all_fixed.csv")

    dt_train = dt
    dt_train.to_csv(os.path.join(args.out_dir, "train.csv"))
    train_MAE_dataset = MAE_dataset(dt_train)
    train_img2img_dataset = img2img_dataset(dt_train)
    train_img2text_dataset = img2text_dataset(dt_train)


    train_datasets = [train_MAE_dataset, train_img2img_dataset, train_img2text_dataset]
    train_dataloader = CustomDataLoader(train_datasets, batch_size=args.batch_size, shuffle=True)
    for d in train_datasets:
        print("dataset:")
        print(len(d))


    #https://github.com/openai/CLIP/issues/57
    def convert_models_to_fp32(model): 
        for p in model.parameters(): 
            p.data = p.data.float() 
            p.grad.data = p.grad.data.float() 

    def convert_models_to_mix(model):
        eyeclip.model.convert_weights(model)
    
    if device == "cpu":
        model.float()
    else :
        eyeclip.model.convert_weights(model)
    
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    train_log_path = os.path.join(args.out_dir, "train.log")
    loss_fig_path = os.path.join(args.out_dir, "train_loss.jpg")
    logging.basicConfig(filename=train_log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    best_valid_loss = float("inf")
    counter = 0

    for epoch in range(args.epochs):
        with tqdm(total=len(train_dataloader)) as epoch_pbar:
            epoch_pbar.set_description(f"Epoch {epoch}")
            start_time = time.time()
            epoch_total_loss = 0.0
            MAE_total_loss = 0.0
            CLIP_total_loss_img = 0.0
            CLIP_total_loss_text = 0.0
            for i, batch in enumerate(train_dataloader):
                flag_img2img=1
                flag_img2txt=1
                lr_sched.adjust_learning_rate(optimizer, i / len(train_dataloader) + epoch, args)
                optimizer.zero_grad()
                MAE_images, img2imgs, img2texts = batch
                MAE_images = MAE_images.to(device)

                ground_truth = torch.arange(len(MAE_images),dtype=torch.long,device=device)
                MAEloss, _, _ = model.visual(MAE_images.to(dtype=model.dtype, device=device), wMAE=True)

                if img2imgs is None:
                    flag_img2img=0
                    CLIP_loss_img = torch.tensor(0.)
                if img2texts is None:
                    flag_img2txt=0
                    CLIP_loss_text = torch.tensor(0.)
                if flag_img2img:
                    images1, images2 = img2imgs
                    images1 = images1.to(device)
                    images2 = images2.to(device)
                    logits_per_image1, logits_per_image2 = model(images1, images2, text=False)
                    CLIP_loss_img = (loss_img(logits_per_image1,ground_truth) + loss_txt(logits_per_image2,ground_truth))/2

                if flag_img2txt:
                    imagest, texts = img2texts
                    imagest = imagest.to(device)
                    texts = texts.to(device)
                    logits_per_image, logits_per_text = model(imagest, texts, text=True)
                    CLIP_loss_text = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2

                total_loss = MAEloss+CLIP_loss_text+CLIP_loss_img

                epoch_total_loss += total_loss.item()
                avg_loss = epoch_total_loss / (i+1)
                total_loss.backward()
                MAE_total_loss += (MAEloss).item()
                CLIP_total_loss_img += (CLIP_loss_img).item()
                CLIP_total_loss_text += (CLIP_loss_text).item()
                if device == "cpu":
                    optimizer.step()
                else : 
                    model = model.float()
                    optimizer.step()
                    convert_models_to_mix(model)
                desc = f"Epoch {epoch} - loss {avg_loss:.20f}"
                epoch_pbar.set_description(desc)
                epoch_pbar.update(images1.shape[0])
                if i%10==0:
                    logging.info(f"Train: Epoch {epoch} - iter {i} - loss {avg_loss:.20f} - MAEloss {(MAE_total_loss/(i+1)):.20f} - CLIPloss_img {(CLIP_total_loss_img/(i+1)):.20f}- CLIPloss_text {(CLIP_total_loss_text/(i+1)):.20f}")

                if i % 1000==0:
                    torch.save({
                    'epoch': str(epoch)+'_'+str(i),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    }, os.path.join(args.out_dir, "epoch"+str(epoch)+'_'+str(i)+".pt"))
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    }, os.path.join(args.out_dir, "epoch"+str(epoch)+".pt"))

        elapsed_time = time.time() - start_time
        plot_training_losses(train_log_path, loss_fig_path)
 