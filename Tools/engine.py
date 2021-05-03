from .base_engine import *

class DA_Engine(BaseEngine):
    """
        Domain Adaptation model for statefarm. 
    """
    def __init__(self, train_loader, eval_loader, test_loader, args, writer, device) -> None:
        super(DA_Engine, self).__init__(train_loader, eval_loader, test_loader, args, writer, device)

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        for i, data in enumerate(self.train_loader):

            p = float(i + epoch * len(self.train_loader)) / self.args.epochs / len(self.train_loader)
            alpha = 2.0 / (1.0 + np.exp(-18*p)) - 1

            img_0, img_1, cls_label, sia_label = data
            img_0 = img_0.to(self.device)
            img_1 = img_1.to(self.device)
            cls_label = cls_label.to(self.device)
            sia_label = sia_label.to(self.device)

            cls_output, sia_output = self.model(img_0, img_1, alpha)
            cls_loss = self.criterion(cls_output, cls_label)
            sia_loss = self.criterion(sia_output, sia_label)
            loss = cls_loss + 0.5*sia_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(self.train_loader)

    def eval_epoch(self, epoch):
        self.model.eval()
        epoch_loss = 0
        for i, data in enumerate(self.eval_loader):
            img_0, img_1, cls_label, sia_label = data
            img_0 = img_0.to(self.device)
            cls_label = cls_label.to(self.device)
            with torch.no_grad():
                cls_output = self.model(img_0, None, None)
            loss = self.criterion(cls_output, cls_label)
            epoch_loss += loss.item()
            self.metric(cls_output, cls_label)
        return epoch_loss / len(self.eval_loader), self.metric.value()