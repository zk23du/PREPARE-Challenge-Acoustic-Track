
from lib import *

class CenterLoss(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int, device):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))  # Centers on the correct device

    def forward(self, features, labels):
        batch_size = features.size(0)

        # Compute pairwise distances between features and centers
        distances = torch.cdist(features, self.centers)

        # Get distances for true class labels
        labels_one_hot = torch.zeros(batch_size, self.num_classes).to(self.device)
        labels_one_hot.scatter_(1, labels.view(-1, 1), 1)
        selected_distances = distances * labels_one_hot
        loss = selected_distances.sum() / batch_size
        return loss

    def predict(self, features):
        # Predict nearest class center
        distances = torch.cdist(features, self.centers)
        predictions = torch.argmin(distances, dim=1)
        return predictions
# classifier = Classifier(input_dim=input_dim, output_dim=output_dim).to(device)
# center_loss = CenterLoss(num_classes=output_dim, feat_dim=128, device=device).to(device)


class AAM_Softmax(nn.Module):
    def __init__(self,
                 number_classes: int = 5994,
                 margin: float = 0.3,
                 scale: int = 12,
                 embedding: int = 512,
                 *args, **kwargs) -> None:
        super().__init__()
        self.scale = scale
        self.weight = torch.nn.Parameter(torch.FloatTensor(number_classes, embedding))
        self.cosine_margin = math.cos(margin)
        self.sin_margin = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, INPUT, label):
        self.weight = self.weight.to(INPUT.device)  # Ensure weight is on the same device
        cosine = F.linear(F.normalize(INPUT), F.normalize(self.weight))
        phi = cosine * self.cosine_margin - torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1)) * self.sin_margin
        phi = torch.where((cosine - self.threshold) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine).to(INPUT.device)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits = logits * self.scale

        loss = F.cross_entropy(logits, label)
        return loss, logits

#criterion = AAM_Softmax(number_classes=output_dim, embedding=1280).to(device)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss
    
#criterion = FocalLoss()