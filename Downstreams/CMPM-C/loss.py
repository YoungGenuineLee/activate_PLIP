import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import os

class Loss_calc(nn.Module):
    def __init__(self, args):
        super(Loss_calc, self).__init__()
        self.epsilon = args.epsilon
        self.W =Parameter(torch.randn(args.feature_size, args.num_classes))
        self.init_weight()
    def init_weight(self):
        nn.init.xavier_uniform_(self.W.data, gain=1)

    def compute_cmpm_loss(self, image_embeddings, text_embeddings, labels, device):
        """
        Cross-Modal Projection Matching Loss(CMPM)
        :param image_embeddings: Tensor with dtype torch.float32
        :param text_embeddings: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:
            i2t_loss: cmpm loss for image projected to text
            t2i_loss: cmpm loss for text projected to image
            pos_avg_sim: average cosine-similarity for positive pairs
            neg_avg_sim: averate cosine-similarity for negative pairs
        """

        batch_size = image_embeddings.shape[0]
        labels_reshape = torch.reshape(labels, (batch_size, 1))
        labels_dist = labels_reshape - labels_reshape.t()
        labels_mask = (labels_dist == 0)

        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        image_proj_text = torch.matmul(image_embeddings, text_norm.t())
        text_proj_image = torch.matmul(text_embeddings, image_norm.t())

        # normalize the true matching distribution
        labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)

        i2t_pred = F.softmax(image_proj_text, dim=1)
        i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + self.epsilon))
        t2i_pred = F.softmax(text_proj_image, dim=1)
        t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + self.epsilon))

        # print("This is compute_Cmpm_loss")
        # print("This is CMPM_pred and loss collections")
        # print(i2t_pred, 1111111)
        # print(i2t_loss, 2222222)
        # print(t2i_pred, 3333333)
        # print(t2i_loss, 4444444)

        cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

        return cmpm_loss

    def compute_cmpc_loss(self, image_embeddings, text_embeddings, labels, device):
        """
        Cross-Modal Projection Classfication loss(CMPC)
        :param image_embeddings: Tensor with dtype torch.float32
        :param text_embeddings: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:
        """
        
        # rank = int(os.environ["RANK"])
        # local_rank = int(os.environ["LOCAL_RANK"])
        # print(local_rank, "This is local_Rank")
        # torch.cuda.set_device(rank % torch.cuda.device_count())
        # device = torch.device("cuda", local_rank)

        labels = labels.type(torch.LongTensor).to(device)

        criterion = nn.CrossEntropyLoss(reduction='mean')
        self.W_norm = self.W / self.W.norm(dim=0)
        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        image_proj_text = torch.sum(image_embeddings * text_norm, dim=1, keepdim=True) * text_norm
        text_proj_image = torch.sum(text_embeddings * image_norm, dim=1, keepdim=True) * image_norm
        image_logits = torch.matmul(image_proj_text, self.W_norm)
        text_logits = torch.matmul(text_proj_image, self.W_norm)
        # 아래가 문제가 되는 녀석임. 정규화가 존나 안된듯?
        # print("This is image_logits")
        # print(image_logits, 'iiiiiiiiiiiiiiiiiiiiiiii')
        # print("This is text_logits")
        # print(text_logits, 'tttttttttttttttttttttttt')
        # print("This is Labels")
        # print(labels, 'lalalalalalallallalallallallalallala')

        cmpc_loss = criterion(image_logits, labels) + criterion(text_logits, labels)
        image_pred = torch.argmax(image_logits, dim=1)
        text_pred = torch.argmax(text_logits, dim=1)
        image_precision = torch.mean((image_pred == labels).float())
        text_precision = torch.mean((text_pred == labels).float())

        # print("This is CMPC Loss")
        # print(cmpc_loss, 5555555)
        # print(image_pred, 6666666)
        # print(text_pred, 7777777)
        # print(image_precision, 8888888)
        # print(text_precision, 9999999)

        return cmpc_loss, image_pred, text_pred, image_precision, text_precision

    def forward(self, global_visual_embed, global_textual_embed, IDlabels, device):
        cmpm_loss = self.compute_cmpm_loss(global_visual_embed,global_textual_embed,IDlabels, device)
        cmpc_loss, image_pred, text_pred, image_precision, text_precision = self.compute_cmpc_loss(global_visual_embed, global_textual_embed,IDlabels, device)
        loss = cmpm_loss + cmpc_loss

        # print("This is Forward Function")
        # print(cmpm_loss, 123123123)
        # print(cmpc_loss, 456456456)
        # print(loss, 78978789)
        # print(image_precision, 1818181818)
        # print(text_precision, 272727272727)
        
        return cmpm_loss, cmpc_loss, loss, image_precision, text_precision