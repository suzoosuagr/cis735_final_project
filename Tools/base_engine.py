import torch
from Tools.logger import *
import torch.optim as optim
import numpy as np

class BaseEngine():
    def __init__(self, train_loader, eval_loader, test_loader, args, writer, device) -> None:
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader
        self.args = args
        self.writer = writer
        self.device = device

    def save_ckpt(self, model, optimizer, epoch, metric):
        """
            save the model ckpt format
        """
        path = self.args.ckpt_root
        name = self.args.name
        ensure(path)

        model_path = os.path.join(path, name+'.pth')
        ckpt = {
            'epoch': epoch,
            'metric': metric,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(ckpt, model_path)

    def load_ckpt(self, model, optimizer=None):
        """
            load the ckpt for resume
        """
        path = self.args.ckpt_root
        name = self.args.name
        model_path = os.path.join(path, name+'.pth')
        assert os.path.isfile(model_path), "The model file {} does not exist."

        ckpt = torch.load(model_path, map_location=self.device)
        model.load_state_dict(ckpt['state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
        epoch = ckpt['epoch']
        metric = ckpt['metric']
        return model, optimizer, epoch, metric

    def train(self, model, optimizer, criterion, metric):
        """
            entry for training
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.min_loss = 1e8
        self.max_accu = 0.0
        start_epoch = 0
        self.metric = metric
        
        info("Train >>> {}".format(self.args.name))

        # resume
        if self.args.resume:
            self.model, self.optimizer, start_epoch, self.min_loss = self.load_ckpt(self.model, self.optimizer)
            info("Resume from {} epoch".format(start_epoch))

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(  self.optimizer, 
                                                                'min',
                                                                patience=self.args.patience,
                                                                min_lr=1e-5
                                                                )
        # EPOCH LOOP
        for epoch in range(self.args.epochs):
            info("Epc: {}/{}, val_loss:{}".format(epoch, self.args.epochs, self.min_loss))
            train_loss = self.train_epoch(epoch)
            # _, _ = self.eval_epoch(epoch)

            try:
                self.writer.add_scalar('train/loss', train_loss, epoch)
            except AttributeError:
                pass
            
            # VALIDATION
            if (epoch+1) % self.args.eval_freq == 0 or epoch+1 == self.args.epochs:
                eval_loss, metric_ = self.eval_epoch(epoch)
                
                try:
                    self.writer.add_scalar('eval/loss', eval_loss, epoch)
                    self.writer.add_scalar('eval/{}'.format(self.metric.__class__.__name__), epoch)
                except AttributeError:
                    pass

                if self.min_loss > eval_loss:
                    self.min_loss = eval_loss
                    info("New min Val loss {:.4f} at [{:03}]: Val_{}:{:.2f}".format(eval_loss, epoch, self.metric.__class__.__name__, metric_))
                    self.save_ckpt(self.model, self.optimizer, epoch, self.min_loss)
                self.scheduler.step(eval_loss)
                info("current lr_rate is {}".format(self.scheduler._last_lr))

    def test(self):
        raise NotImplementedError

    def train_epoch(self, epoch):
        raise NotImplementedError
    
    def eval_epoch(self, epoch):
        raise NotImplementedError



        


