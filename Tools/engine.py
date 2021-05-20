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

    def valid(self, model, optimizer, criterion, metric):
        self.model, _, start_epoch, self.min_loss = self.load_ckpt(model, optimizer)
        info("Test resume from {} epoch".format(start_epoch))
        self.model.eval()
        epoch_loss = 0
        for i, data in enumerate(self.eval_loader):
            img_0, img_1, cls_label, sia_label = data
            img_0 = img_0.to(self.device)
            cls_label = cls_label.to(self.device)
            with torch.no_grad():
                cls_output = self.model(img_0, None, None)
            loss = criterion(cls_output, cls_label)
            epoch_loss += loss.item()
            metric(cls_output, cls_label)
        return epoch_loss / len(self.eval_loader), metric.value()

    def submission(self, model, optimizer, submission_path):
        rounder = lambda x: round(x, 4)
        self.model, _, start_epoch, self.min_loss = self.load_ckpt(model, optimizer)
        info("Test resume from {} epoch".format(start_epoch))
        self.model.eval()
        sub_writer = open(submission_path, 'w') 
        sub_writer.write("img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n")
        for i, data in enumerate(self.test_loader):
            img_0, img_pathes = data
            img_0 = img_0.to(self.device)
            with torch.no_grad():
                cls_output = self.model(img_0, None, None)
            cls_preds = torch.softmax(cls_output, dim=1)
            for i in range(len(cls_output)):
                msg = os.path.basename(img_pathes[i])+','
                msg += ','.join([str(rounder(p)) for p in cls_preds[i].cpu().numpy().tolist()])+'\n'
                sub_writer.write(msg)
        sub_writer.close()
        warning("Generated submission file at {}".format(submission_path))






class Siamese_Engine(BaseEngine):
    def __init__(self, train_loader, eval_loader, test_loader, args, writer, device) -> None:
        super(Siamese_Engine, self).__init__(train_loader, eval_loader, test_loader, args, writer, device)

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        for i, data in enumerate(self.train_loader):
            # reverse layer. 
            p = float(i + epoch * len(self.train_loader)) / self.args.epochs / len(self.train_loader)
            alpha = 2.0 / (1.0 + np.exp(-18*p)) - 1

            img_0, img_1, cls_label, sia_label = data # sia_label: 0 similar person, 1 dissimilar person.
            img_0 = img_0.to(self.device)
            img_1 = img_1.to(self.device)
            cls_label = cls_label.to(self.device)
            sia_label = sia_label.to(self.device)

            cls_output, feat0, feat1 = self.model(img_0, img_1, alpha)
            cls_loss = self.criterion['cls'](cls_output, cls_label)
            rev_contras_loss = self.criterion['rev_sia'](feat0, feat1, sia_label)
            loss = cls_loss + 0.01*rev_contras_loss

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
            loss = self.criterion['cls'](cls_output, cls_label)
            epoch_loss += loss.item()
            self.metric(cls_output, cls_label)
        return epoch_loss / len(self.eval_loader), self.metric.value()





