import torch
import numpy as np
import torch.nn.functional as F

class Metric():
    """Base class for all metrics. 
    """
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class Performance_Test(Metric):
    def __init__(self, size=3):
        """
        For single label\\
        `size`: int, the size of confusion matrix, the total number of class
        """
        super(Performance_Test, self).__init__()
        self.confusion_matrix = np.zeros((size, size))
        self.size = size

    def __call__(self, pred, truth):
        pred = torch.argmax(pred, dim=-1).long()
        for p, t in zip(pred, truth):
            self.confusion_matrix[t][p] += 1

    def reset(self):
        self.confusion_matrix = np.zeros((self.size, self.size))

    def value(self):
        """
        calculate the precision and recall. 
        https://en.wikipedia.org/wiki/Confusion_matrix
        
        """
        TP = self.confusion_matrix.diagonal()
        precision = TP / self.confusion_matrix.sum(0)
        recall = TP / self.confusion_matrix.sum(1)

        f1 = (2*precision*recall) / (precision + recall)

        accu = TP.sum()
        accu = TP.sum() / self.confusion_matrix.sum()

        return precision, recall, f1, accu, self.confusion_matrix


class Accu(Metric):
    def __init__(self, ):
        super(Accu, self).__init__()
        self.correct = 0
        self.total = 0
    def __call__(self, pred, truth):
        pred = torch.argmax(pred, dim=-1).long()
        self.correct += torch.sum(pred==truth)
        self.total += len(pred)

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return self.correct / (self.total + 1e-8) 