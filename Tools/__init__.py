from .utils import *
import random
import torch
import numpy as np
from .logger import *
from torch.utils.tensorboard import SummaryWriter
import datetime

# initializer
def env_init(args, loglevel=logging.INFO):
    # fix seeds. 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # select device
    setup_global_logger(args.mode, loglevel, logpath=args.logpath)

    device_ = 'cpu'
    if torch.cuda.is_available():
        gpu = auto_select_gpu(utility_bound=0, num_gpu=torch.cuda.device_count())
        device_ = 'cuda'
        info("DEVICE: {}, found {} gpus: {}".format(device_, len(gpu), gpu))
    else:
        warning("No gpu found, using cpu instead")
    device = torch.device(device_)

    # summary writer
    if args.summary:
        if args.mode in args.summary_register:
            if not os.path.isdir(args.summary_dir):
                os.mkdir(args.summary_dir)
            summary_dir = os.path.join(args.summary_dir, args.name+'/',datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))

            writer = SummaryWriter(summary_dir)
            info("tfboard writer created. ")
        else:
            writer = None
    else:
        writer = None

    return len(gpu), device, writer


    