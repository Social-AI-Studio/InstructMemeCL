
import torch
import torch.nn as nn

class Cluster_Loss(nn.Module):
    """
    Cluster loss function.
    Based on:
    """

    def __init__(self, margin=1.0, metric = 'l2'):
        super(Cluster_Loss, self).__init__()
        self.margin = margin
        self.metric = metric
        # print('ContrastiveLoss, Metric:', self.metric)

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y):
        #self.check_type_forward((x0, x1, y))

        # euclidian distance
        if self.metric == 'l2':
            diff = x0 - x1
            dist_sq = torch.sum(torch.pow(diff, 2), -1) / x0.shape[-1]
            if torch.any(torch.isnan(dist_sq)):
                print("nan error")
            dist = torch.sqrt(dist_sq)
            if torch.any(torch.isnan(dist)):
                print("nan error")
        elif self.metric == 'cos':
            # import pdb
            # pdb.set_trace()
            # prod = torch.sum(x0 * x1, -1)
            # if torch.any(torch.isnan(prod)):
            #     import pdb
            #     pdb.set_trace()
            #     print("nan error")
            # dist = 1 - prob /  torch.maximum(torch.sqrt(torch.sum(x0**2, 1) * torch.sum(x1**2, 1)),(torch.ones(367)*1e-8).cuda()) #dissimilarity
            # torch.sqrt(torch.sum(x0**2, 1)) * torch.sqrt(torch.sum(x1**2, 1))
            # if torch.any(torch.isnan(dist)):
            #     import pdb
            #     pdb.set_trace()
            #     print("nan error")
            # dist_sq = dist ** 2
            dist = 1 - torch.cosine_similarity(x0, x1, dim=-1)
            dist_sq = dist ** 2
            
            
            #print(x0, x1, torch.sum(torch.pow(x0-x1, 2), 1) / x0.shape[-1], dist, dist_sq)
        else:
            print("Error Loss Metric!!")
            return 0
        #dist = torch.sum( - x0 * x1 / np.sqrt(x0.shape[-1]), 1).exp()
        #dist_sq = dist ** 2

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        if dist.dim() == 1:
            loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        elif dist.dim() == 2:
            _ , seq_len = dist.shape
            loss = y.unsqueeze(-1).expand(-1, seq_len) * dist_sq + (1 - y).unsqueeze(-1).expand(-1, seq_len) * torch.pow(dist, 2)
        else:
            raise KeyError
        if torch.any(torch.isnan(loss)):
            print("nan error")
        loss = torch.sum(loss, dim=-1) / 2.0 / x0.size()[0]
        if torch.any(torch.isnan(loss)):
            print("nan error")
        return loss #, dist_sq, dist

class Cluster_Loss_1(nn.Module):
    """
    Cluster loss function.
    Based on:
    """

    def __init__(self, margin=1.0):
        super(Cluster_Loss_1, self).__init__()
        self.margin = margin
        # print('ContrastiveLoss, Metric:', self.metric)

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, pred_score, y):
        #self.check_type_forward((x0, x1, y))

        dist = 1 - pred_score
        dist_sq = dist ** 2
            
        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        if dist.dim() == 1:
            loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        elif dist.dim() == 2:
            _ , seq_len = dist.shape
            loss = y.unsqueeze(-1).expand(-1, seq_len) * dist_sq + (1 - y).unsqueeze(-1).expand(-1, seq_len) * torch.pow(dist, 2)
        else:
            raise KeyError
        if torch.any(torch.isnan(loss)):
            print("nan error")
        loss = torch.sum(loss, dim=-1) / 2.0 / pred_score.size()[0]
        if torch.any(torch.isnan(loss)):
            print("nan error")
        return loss #, dist_sq, dist

class Triplet_Loss(nn.Module):
    """
    Triplet loss function.
    Based on:
    """

    def __init__(self, margin=1.0, metric = 'l2'):
        super(Triplet_Loss, self).__init__()
        self.margin = margin
        self.metric = metric
        # print('ContrastiveLoss, Metric:', self.metric)

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, anchor, positive, negative):
        #self.check_type_forward((x0, x1, y))

        if self.metric == "l2":
            criterion = torch.nn.TripletMarginLoss(self.margin, reduction = 'mean')
        elif self.metric == "cos":
            criterion = torch.nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - torch.cosine_similarity(x, y), margin=self.margin, reduction = 'mean')
        else:
            print("Error Loss Metric!!")
            return 0
        
        loss = criterion(anchor, positive, negative)
        return loss 

