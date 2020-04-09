import torch
import torch.nn as nn
import torch.nn.functional as F

class TextLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def ohem(self, predict, target, train_mask, negative_ratio=3.):
        pos = (target * train_mask).byte() # get background
        if pos.sum() == 0:
            pos = (target).byte()
        neg = ((1 - target) * train_mask).byte() # get the area that don't ignored

        n_pos = pos.float().sum()
        n_neg = min(int(neg.float().sum().item()), int(negative_ratio * n_pos.float()))

        loss_pos = F.cross_entropy(predict[pos], target[pos], size_average=False)
        loss_neg = F.cross_entropy(predict[neg], target[neg], reduce=False)
        loss_neg, _ = torch.topk(loss_neg, n_neg)  # get the bigger

        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).float()

    def ohem_rectangular_box(self, predict, target, negative_ratio=3.):
        pos = (target ).byte() # get background
        neg = (1 - target).byte() # get the area that don't ignored

        n_pos = pos.float().sum()
        n_neg = min(int(neg.float().sum().item()), int(negative_ratio * n_pos.float()))


        loss_pos = F.cross_entropy(predict[pos], target[pos], size_average=False)
        loss_neg = F.cross_entropy(predict[neg], target[neg], reduce=False)
        loss_neg, _ = torch.topk(loss_neg, n_neg)  # get the bigger

        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).float()

    def forward(self, input, tr_mask, tcl_mask, tcl_weight,up_mask, down_mask, left_mask, right_mask, train_mask):
        """
        calculate textsnake loss
        :param input: (Variable), network predict, (BS, 7, H, W)
        :param tr_mask: (Variable), TR target, (BS, H, W)
        :param tcl_mask: (Variable), TCL target, (BS, H, W)
        :param sin_map: (Variable), sin target, (BS, H, W)
        :param cos_map: (Variable), cos target, (BS, H, W)
        :param radii_map: (Variable), radius target, (BS, H, W)
        :param train_mask: (Variable), training mask, (BS, H, W)
        :return: loss_tr, loss_tcl, loss_radii, loss_sin, loss_cos
        """

        tr_pred = input[:, :2].permute(0, 2, 3, 1).contiguous().view(-1, 2)  # (BSxHxW, 2)
        tcl_pred = input[:, 2:4].permute(0, 2, 3, 1).contiguous().view(-1, 2)  # (BSxHxW, 2)

        up_pred = input[:, 4].contiguous().view(-1)
        down_pred = input[:, 5].contiguous().view(-1)
        left_pred = input[:, 6].contiguous().view(-1)
        right_pred = input[:, 7].contiguous().view(-1)

        rectangular_box_pred = input[:, 8].contiguous().view(-1)


        train_mask = train_mask.contiguous().view(-1)  # (BSxHxW,)

        tr_mask = tr_mask.contiguous().view(-1)

        tcl_mask = tcl_mask.contiguous().view(-1)
        tcl_weight = tcl_weight.contiguous().view(-1)

        # four direction
        up_mask = up_mask.contiguous().view(-1)
        down_mask = down_mask.contiguous().view(-1)
        right_mask = right_mask.contiguous().view(-1)
        left_mask = left_mask.contiguous().view(-1)


        # tr loss computation
        loss_tr = self.ohem(tr_pred, tr_mask.long(), train_mask.long())

        # tcl loss compuation
        try:
            loss_tcl = F.cross_entropy(tcl_pred[train_mask * tr_mask], \
                                      tcl_mask[train_mask * tr_mask].long(),reduce=False)
            tcl_weight = tcl_weight[train_mask * tr_mask].float()
        except:
            loss_tcl = F.cross_entropy(tcl_pred[tr_mask], \
                                       tcl_mask[tr_mask].long(), reduce=False)
            tcl_weight = tcl_weight[tr_mask].float()
        
        loss_tcl = (tcl_weight * loss_tcl).float()
        loss_tcl = torch.mean(loss_tcl)

        # four directions loss computation
        loss_up = F.smooth_l1_loss(up_pred,up_mask)
        loss_down = F.smooth_l1_loss(down_pred,down_mask)
        loss_left = F.smooth_l1_loss(left_pred,left_mask)
        loss_right =F.smooth_l1_loss(right_pred,right_mask)

        # loss_rectangular_box = F.smooth_l1_loss(rectangular_box_pred, rectangular_box)

        return loss_tr, loss_tcl, loss_up, loss_down, loss_left, loss_right