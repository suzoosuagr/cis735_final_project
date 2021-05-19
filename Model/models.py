from .networks import *
import torchvision.models as models

class DANN_resnet34(nn.Module):
    def __init__(self, nclass, pretrained) -> None:
        super(DANN_resnet34, self).__init__()
        backbone = models.resnet34(pretrained)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        fc_in_dim = list(backbone.children())[-1].in_features
        self.classifier = MLP_Classifier(fc_in_dim, nclass)                  # 512, 
        self.domain = MLP_Classifier(fc_in_dim*2, 2)

    def forward(self, img_0, img_1, alpha):
        if alpha is None:
            return self.cls_forward(img_0)

        feat_0 = self.features(img_0).squeeze(2).squeeze(2)
        feat_1 = self.features(img_1).squeeze(2).squeeze(2)

        class_output = self.classifier(feat_0)
        sia_feat = torch.cat([feat_0, feat_1], dim=-1)
        reverse_feat = ReverseLayerF.apply(sia_feat, alpha)

        domain_output = self.domain(reverse_feat)
        return class_output, domain_output

    def cls_forward(self, img_0):
        feat_0 = self.features(img_0).squeeze(2).squeeze(2)
        class_output = self.classifier(feat_0)
        return class_output

class RevSiamese_resnet34(nn.Module):
    def __init__(self, nclass, pretrained) -> None:
        super(RevSiamese_resnet34, self).__init__()
        backbone = models.resnet34(pretrained)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        fc_in_dim = list(backbone.children())[-1].in_features
        self.classifier = nn.Linear(fc_in_dim, nclass)

    def forward(self, img_0, img_1, alpha):
        if alpha is None:
            return self.cls_forward(img_0)

        feat_0 = self.features(img_0).squeeze(2).squeeze(2)
        feat_1 = self.features(img_1).squeeze(2).squeeze(2)

        class_output = self.classifier(feat_0)
        # reverse_feat0 = ReverseLayerF.apply(sia_feat, alpha)

        return class_output, feat_0, feat_1

    def cls_forward(self, img_0):
        feat_0 = self.features(img_0).squeeze(2).squeeze(2)
        class_output = self.classifier(feat_0)
        return class_output