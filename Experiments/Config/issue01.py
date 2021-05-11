class Basic_Config():
    def __init__(self) -> None:
        self.seed = 2333
        self.issue = __file__.split('/')[-1].split('.')[0].upper()  # using the this filename as issue value.
        self.device = 'cuda'
        self.summary_dir = './TFrecords/'
        self.log_root = './Log/'
        self.summary_register = ['train']

class EXP1(Basic_Config):
    """
        Domain adaptation, cross subjects. 
    """
    def __init__(self, mode, logfile) -> None:
        super(EXP1, self).__init__()
        self.mode = mode
        self.expid = self.__class__.__name__
        self.name = self.issue + '_' + self.expid
        self.logfile = logfile
        self.ckpt_root = '../model_weights/cis_735_final/'
        self.ins_train_file = './Dataset/instruction/statefarm_train.txt'
        self.ins_val_file = './Dataset/instruction/statefarm_val.txt' 
        self.img_size = 224

        self.summary = False
        self.resume = False

        self.backbone = 'res34'
        self.nclass = 10
        self.lr = 1e-3
        self.optim = 'Adam'
        self.weight_decay = 5e-5
        self.batch = 1024
        self.eval_freq = 1
        self.epochs = 200
        self.patience = 10
        self.alpha = 0.4
        self.num_workers = 16
        self.pin_memory = True

        
