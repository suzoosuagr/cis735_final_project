from .networks import *
import torchvision.models as models

class DANN_resnet34(nn.Module):
    def __init__(self, nclass, pretrained) -> None:
        super(DANN_resnet34, self).__init__()
        backbone = models.resnet34(pretrained)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        fc_in_dim = list(backbone.children())[-1].in_features
        self.classifier = nn.Linear(fc_in_dim, nclass)
        self.siamese = nn.Linear(fc_in_dim*2, 2)

    def forward(self, img_0, img_1, alpha):
        if alpha is None:
            return self.cls_forward(img_0)

        feat_0 = self.features(img_0).squeeze(2).squeeze(2)
        feat_1 = self.features(img_1).squeeze(2).squeeze(2)

        class_output = self.classifier(feat_0)
        sia_feat = torch.cat([feat_0, feat_1], dim=-1)
        reverse_feat = ReverseLayerF.apply(sia_feat, alpha)

        sia_output = self.siamese(reverse_feat)
        return class_output, sia_output

    def cls_forward(self, img_0):
        feat_0 = self.features(img_0).squeeze(2).squeeze(2)
        class_output = self.classifier(feat_0)
        return class_output


class DANN_resnet50(nn.Module):
    def __init__(self, nclass, pretrained) -> None:
        super(DANN_resnet34, self).__init__()
        backbone = models.resnet50(pretrained)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        fc_in_dim = list(backbone.children())[-1].in_features
        self.classifier = nn.Linear(fc_in_dim, nclass)
        self.domain = nn.Linear(fc_in_dim*2, 2)

    def forward(self, img_0, img_1, alpha):
        if alpha is None:
            return self.cls_forward(img_0)

        feat_0 = self.features(img_0).squeeze(2).squeeze(2)
        feat_1 = self.features(img_1).squeeze(2).squeeze(2)

        class_output = self.classifier(feat_0)
        sia_feat = torch.cat([feat_0, feat_1], dim=-1)
        reverse_feat = ReverseLayerF.apply(sia_feat, alpha)

        sia_output = self.siamese(reverse_feat)
        return class_output, sia_output

    def cls_forward(self, img_0):
        feat_0 = self.features(img_0).squeeze(2).squeeze(2)
        class_output = self.classifier(feat_0)
        return class_output