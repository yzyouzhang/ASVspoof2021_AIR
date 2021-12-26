import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
import numpy as np


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunction = CenterlossFunction.apply

    def forward(self, feat, y):
        # To squeeze the Tenosr
        batch_size = feat.size(0)
        feat = feat.view(batch_size, 1, 1, -1).squeeze()
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's dim: {1}".format(self.feat_dim,feat.size(1)))
        return self.centerlossfunction(feat, y, self.centers)


class CenterlossFunction(Function):

    @staticmethod
    def forward(ctx, feature, label, centers):
        ctx.save_for_backward(feature, label, centers)
        centers_pred = centers.index_select(0, label.long())
        return (feature - centers_pred).pow(2).sum(1).sum(0) / 2.0


    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers = ctx.saved_variables
        grad_feature = feature - centers.index_select(0, label.long()) # Eq. 3

        # init every iteration
        counts = torch.ones(centers.size(0))
        grad_centers = torch.zeros(centers.size())
        if feature.is_cuda:
            counts = counts.cuda()
            grad_centers = grad_centers.cuda()
        # print counts, grad_centers

        # Eq. 4 || need optimization !! To be vectorized, but how?
        for i in range(feature.size(0)):
            # j = int(label[i].data[0])
            j = int(label[i].item())
            counts[j] += 1
            grad_centers[j] += (centers.data[j] - feature.data[i])
        # print counts
        grad_centers = Variable(grad_centers/counts.view(-1, 1))

        return grad_feature * grad_output, None, grad_centers


class AngularIsoLoss(nn.Module):
    def __init__(self, feat_dim=2, r_real=0.9, r_fake=0.5, alpha=20.0):
        super(AngularIsoLoss, self).__init__()
        self.feat_dim = feat_dim
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.center = nn.Parameter(torch.randn(1, self.feat_dim))
        nn.init.kaiming_uniform_(self.center, 0.25)
        self.softplus = nn.Softplus()

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        scores = x @ w.transpose(0,1)
        output_scores = scores.clone()

        scores[labels == 0] = self.r_real - scores[labels == 0]
        scores[labels == 1] = scores[labels == 1] - self.r_fake

        # loss = self.softplus(torch.logsumexp(self.alpha * scores, dim=0))

        # loss = self.softplus(torch.logsumexp(self.alpha * scores[labels == 0], dim=0)) + \
        #        self.softplus(torch.logsumexp(self.alpha * scores[labels == 1], dim=0))

        loss = self.softplus(self.alpha * scores).mean()

        # print(output_scores.squeeze(1).shape)

        return loss, -output_scores.squeeze(1)

class IsolateLoss(nn.Module):
    """Isolate loss.

        Reference:
        I. Masi, A. Killekar, R. M. Mascarenhas, S. P. Gurudatt, and W. AbdAlmageed, “Two-branch Recurrent Network for Isolating Deepfakes in Videos,” 2020, [Online]. Available: http://arxiv.org/abs/2008.03412.
        Args:
            num_classes (int): number of classes.
            feat_dim (int): feature dimension.
        """
    def __init__(self, num_classes=10, feat_dim=2, r_real=0.042, r_fake=1.638):
        super(IsolateLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.r_real = r_real
        self.r_fake = r_fake

        self.center = nn.Parameter(torch.randn(1, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        # batch_size = x.size(0)
        # o1 = nn.ReLU()(torch.norm(x-self.center, p=2, dim=1) - self.r_real).unsqueeze(1)
        # o2 = nn.ReLU()(self.r_fake - torch.norm(x-self.center, p=2, dim=1)).unsqueeze(1)
        #
        # distmat = torch.cat((o1, o2), dim=1)
        #
        # classes = torch.arange(self.num_classes).long().cuda()
        # # classes = classes.cuda()
        # labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        # mask = labels.eq(classes.expand(batch_size, self.num_classes))
        #
        # dist = distmat * mask.float()
        # loss = dist.clamp(min=1e-12, max=1e+12).sum(0) / mask.sum(0)
        # loss = loss.sum()
        loss = F.relu(torch.norm(x[labels==0]-self.center, p=2, dim=1) - self.r_real).mean() \
               + F.relu(self.r_fake - torch.norm(x[labels==1]-self.center, p=2, dim=1)).mean()
        return loss

class IsolateSquareLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2, r_real=0.042, r_fake=1.638):
        super(IsolateSquareLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.r_real = r_real
        self.r_fake = r_fake

        self.center = nn.Parameter(torch.randn(1, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        # batch_size = x.size(0)
        # o1 = nn.ReLU()(torch.norm(x-self.center, p=2, dim=1) - self.r_real).unsqueeze(1)
        # o2 = nn.ReLU()(self.r_fake - torch.norm(x-self.center, p=2, dim=1)).unsqueeze(1)
        #
        # distmat = torch.cat((o1, o2), dim=1)
        #
        # classes = torch.arange(self.num_classes).long().cuda()
        # # classes = classes.cuda()
        # labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        # mask = labels.eq(classes.expand(batch_size, self.num_classes))
        #
        # dist = distmat * mask.float()
        # loss = dist.clamp(min=1e-12, max=1e+12).sum(0) / mask.sum(0)
        # loss = loss.sum()
        loss = F.relu(torch.pow(torch.norm(x[labels==0]-self.center, p=2, dim=1),2) - self.r_real**2).mean() \
               + F.relu(self.r_fake**2 - torch.pow(torch.norm(x[labels==1]-self.center, p=2, dim=1),2)).mean()
        return loss


class OCSoftmax(nn.Module):
    def __init__(self, feat_dim=2, r_real=0.9, r_fake=0.5, alpha=20.0):
        super(OCSoftmax, self).__init__()
        self.feat_dim = feat_dim
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.center = nn.Parameter(torch.randn(1, self.feat_dim))
        nn.init.kaiming_uniform_(self.center, 0.25)
        self.softplus = nn.Softplus()

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        scores = x @ w.transpose(0,1)
        output_scores = scores.clone()

        scores[labels == 0] = self.r_real - scores[labels == 0]
        scores[labels == 1] = scores[labels == 1] - self.r_fake

        loss = self.softplus(self.alpha * scores).mean()

        # print(output_scores.squeeze(1).shape)

        return loss, -output_scores.squeeze(1)


class AMSoftmax(nn.Module):
    def __init__(self, num_classes, enc_dim, s=20, m=0.9):
        super(AMSoftmax, self).__init__()
        self.enc_dim = enc_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.centers = nn.Parameter(torch.randn(num_classes, enc_dim))

    def forward(self, feat, label):
        batch_size = feat.shape[0]
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cuda()
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.m)
        margin_logits = self.s * (logits - y_onehot)
        # print(margin_logits.shape)

        return logits, margin_logits

###################
"""
P2SGrad-MSE
__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"

"""
class P2SGradLoss(nn.Module):
    """ Output layer that produces cos theta between activation vector x
    and class vector w_j

    in_dim:     dimension of input feature vectors
    output_dim: dimension of output feature vectors 
                (i.e., number of classes)
    
    Usage example:
      batchsize = 64
      input_dim = 10
      class_num = 5

      l_layer = P2SActivationLayer(input_dim, class_num)
      l_loss = P2SGradLoss()

      data = torch.rand(batchsize, input_dim, requires_grad=True)
      target = (torch.rand(batchsize) * class_num).clamp(0, class_num-1)
      target = target.to(torch.long)

      scores = l_layer(data)
      loss = l_loss(scores, target)

      loss.backward()
    """
    def __init__(self, in_dim, out_dim, smooth=0.1):
        super(P2SGradLoss, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.smooth = smooth
        
        self.weight = Parameter(torch.Tensor(in_dim, out_dim))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)

        self.m_loss = nn.MSELoss()

    def smooth_labels(self, labels):
        factor = self.smooth

        # smooth the labels
        labels *= (1 - factor)
        labels += (factor / labels.shape[1])
        return labels

    def forward(self, input_feat, target):
        """
        Compute P2sgrad activation
        
        input:
        ------
          input_feat: tensor (batchsize, input_dim)

        output:
        -------
          tensor (batchsize, output_dim)
          
        """
        # normalize the weight (again)
        # w (feature_dim, output_dim)
        w = self.weight.renorm(2, 1, 1e-5).mul(1e5)
        
        # normalize the input feature vector
        # x_modulus (batchsize)
        # sum input -> x_modules in shape (batchsize)
        x_modulus = input_feat.pow(2).sum(1).pow(0.5)
        # w_modules (output_dim)
        # w_moduls should be 1, since w has been normalized
        w_modulus = w.pow(2).sum(0).pow(0.5)

        # W * x = ||W|| * ||x|| * cos())))))))
        # inner_wx (batchsize, output_dim)
        inner_wx = input_feat.mm(w)
        # cos_theta (batchsize, output_dim)
        cos_theta = inner_wx / x_modulus.view(-1, 1)
        cos_theta = cos_theta.clamp(-1, 1)

        # P2Sgrad MSE
        # target (batchsize, 1)
        target = target.long() #.view(-1, 1)
        
        # filling in the target
        # index (batchsize, class_num)
        with torch.no_grad():
            index = torch.zeros_like(cos_theta)
            # index[i][target[i][j]] = 1
            index.scatter_(1, target.data.view(-1, 1), 1)
            index = self.smooth_labels(index)
    
        # MSE between \cos\theta and one-hot vectors
        loss = self.m_loss(cos_theta, index)
        # print(cos_theta[:, 0].shape)
        return loss, -cos_theta[:, 0]


# PyTorch implementation of Focal Loss
# source: https://github.com/clcarwin/focal_loss_pytorch

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):

        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int, long)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

if __name__ == "__main__":
    # feats = torch.randn((32, 90)).cuda()
    # centers = torch.randn((3,90)).cuda()
    # # o = torch.norm(feats - center, p=2, dim=1)
    # # print(o.shape)
    # # dist = torch.cat((o, o), dim=1)
    # # print(dist.shape)
    # labels = torch.cat((torch.Tensor([0]).repeat(10),
    #                    torch.Tensor([1]).repeat(22)),0).cuda()
    # # classes = torch.arange(2).long().cuda()
    # # labels = labels.expand(32, 2)
    # # print(labels)
    # # mask = labels.eq(classes.expand(32, 2))
    # # print(mask)
    #
    # iso_loss = MultiCenterIsolateLoss(centers, 2, 90).cuda()
    # loss = iso_loss(feats, labels)
    # for p in iso_loss.parameters():
    #     print(p)
    # # print(loss.shape)

    feat_dim = 16
    feats = torch.randn((32, feat_dim))
    labels = torch.cat((torch.Tensor([0]).repeat(10),
                        torch.Tensor([1]).repeat(22)), 0)
    aisoloss = P2SGradLoss(in_dim=feat_dim, out_dim=2)
    loss = aisoloss(feats, labels)
    print(loss)