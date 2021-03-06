from Tools import env_init
from Tools.logger import *
import argparse
from Experiments.Config.issue01 import *
from Dataset import StateFarm, StateFarm_Test
from torch.utils.data import DataLoader
from torchvision import transforms as T
import skimage.io as io
from Model.models import DANN_resnet34, RevSiamese_resnet34
import torch.nn as nn
import torch.optim as optim
from Tools.metric import Accu
from Tools.engine import DA_Engine, Siamese_Engine
from Model.networks import Rev_ContrastiveLoss, ContrastiveLoss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", dest="mode", default="debug", type=str)
    parser.add_argument("-l", "--log", dest="logfile", default='debug.log', type=str)
    parser.add_argument('-e',  "--exp", dest="expid", default='EXP1', choices=['EXP1', 'EXP2'])
    
    return parser.parse_args()

# initialization
parser = parse_args()
args = eval(parser.expid)(parser.mode, parser.logfile)
warning("STARTING >>>>>> {} ".format(args.name))
args.logpath = os.path.join(args.log_root, args.name, args.logfile)
ngpu, device, writer = env_init(args, logging.INFO)
args.ngpu = ngpu


# dataset
train_trans = T.Compose(
    [
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(args.img_size),
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        T.RandomRotation(degrees=30, resample=False, expand=False),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_trans = T.Compose(
    [
        T.Resize(args.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)
train_dataset = StateFarm(args.ins_train_file, train_trans)
eval_dataset = StateFarm(args.ins_val_file, val_trans)
test_dataset = StateFarm_Test(args.ins_test_file, val_trans)
info("{} samples for training --- {} samples for validation".format(len(train_dataset), len(eval_dataset)))
# img, label = train_dataset[0]
# io.imsave('./temp/dataset_visual.png', img.permute(1, 2, 0))

if args.mode == 'debug':
    args.pin_memory=False
    args.num_workers=0
    args.batch = 128
    args.resume = False
    args.summary = False

train_loader = DataLoader(  train_dataset, batch_size=args.batch, shuffle=True, \
                            drop_last=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
eval_loader = DataLoader(   eval_dataset, batch_size=args.batch, shuffle=False, \
                            drop_last=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
test_loader = DataLoader(   test_dataset, batch_size=args.batch, shuffle=False, \
                            drop_last=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
                
# model
if args.method == 'DA':
    model = DANN_resnet34(args.nclass, True).to(device)
elif args.method in ['RevSiamese', 'Siamese']:
    model = RevSiamese_resnet34(args.nclass, True).to(device)
model = nn.DataParallel(model, device_ids=range(args.ngpu))
# optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# metric
if args.method == 'DA':
    criterion = nn.CrossEntropyLoss()
elif args.method == 'RevSiamese':
    criterion = {
        'rev_sia': Rev_ContrastiveLoss(margin=args.margin),
        'cls': nn.CrossEntropyLoss()
    }
elif args.method == 'Siamese':
    criterion = {
        'rev_sia': ContrastiveLoss(margin=args.margin),
        'cls':nn.CrossEntropyLoss()
    }
metric = Accu()
# engine
# engine = Siamese_Engine(train_loader, eval_loader, None, args, writer, device)
engine = DA_Engine(train_loader, eval_loader, test_loader, args, writer, device)

# random seed
if __name__ == '__main__':
    if args.mode in ['train']:
        info("Margin = {}".format(args.margin))
        engine.train(model, optimizer, criterion, metric)
        
    elif args.mode in ['valid']:
        valid_loss, valid_accu = engine.valid(model, optimizer, criterion, metric)
        info(f"loss={valid_loss}, accu={valid_accu}")

    elif args.mode in ['submit']:
        sub_file = './submission_{}.csv'.format(args.name)
        engine.submission(model, optimizer, submission_path=sub_file)
