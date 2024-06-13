import torch
from torch import nn


class NCELoss(nn.Module):
    """
    Compute the PointInfoNCE loss
    """

    def __init__(self, temperature):
        super(NCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, k, q, similarity=None, min_quantile=0.0, max_quantile=1.0):
        if similarity is None:
            # compute logits
            logits = torch.mm(k, q.transpose(1, 0))
            target = torch.arange(k.shape[0], device=k.device).long()
            out = torch.div(logits, self.temperature)
        else:
            similarity = torch.clip(similarity, min=0.0, max=1.0)

            # remove superpoints and superpixels that have zero pretrained feature vectors
            nonzero_superpixel_feat_mask = torch.where(torch.sum(similarity, dim=1)!= 0.0)
            # filter similarity where rows and columns are zeros
            similarity = torch.index_select(similarity, dim=0, index=nonzero_superpixel_feat_mask[0])
            similarity = torch.index_select(similarity, dim=1, index=nonzero_superpixel_feat_mask[0])
            # filter superpoints/superpixels
            k = k[nonzero_superpixel_feat_mask]
            q = q[nonzero_superpixel_feat_mask]

            # compute weight of each negative sample based on similarity matrix
            # samples in between postive and negative
            weight = 1.0 - similarity
            # diagonal elements are positive
            if min_quantile > 0.0 or max_quantile < 1.0:
                min_quantile_value = torch.quantile(weight ,q=min_quantile, dim=1)
                max_quantile_value = torch.quantile(weight ,q=max_quantile, dim=1)
                weight[weight<min_quantile_value.view(-1, 1)] = 0.0
                weight[weight>max_quantile_value.view(-1, 1)] = 0.0
                non_zero_indices = torch.nonzero(weight, as_tuple=True)
                weight[non_zero_indices] = 1.0
            
            weight.fill_diagonal_(1.0)

            # compute logits
            logits = torch.mm(k, q.transpose(1, 0))
            target = torch.arange(k.shape[0], device=k.device).long()
            out = torch.div(logits, self.temperature)
            # element-wise multiplication to reduce contribution of negative samples that are partially positive
            out = out * weight


        out = out.contiguous()
        loss = self.criterion(out, target)
        return loss
