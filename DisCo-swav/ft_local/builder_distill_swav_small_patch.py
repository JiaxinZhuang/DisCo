# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, teacher_encoder, dim=128, K=65536, m=0.999, T=0.07, swav_mlp=2048,
                 temp=1e-4, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        temp: temperature for distillation of teacher (default: 1e-4)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.D = dim
        self.Temp = temp

        # student encoder
        self.encoder_q = base_encoder(num_classes=dim)

        # teacher encoder, resnet50w5 does not use BN
        self.encoder_k = teacher_encoder(
            normalize=True,
            hidden_mlp=swav_mlp,
            output_dim=dim,
            batch_norm=not str(teacher_encoder).__contains__('resnet50w5'),
        )

        if mlp:
            # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                              # nn.BatchNorm1d(dim_mlp),
                                              nn.ReLU(),
                                              self.encoder_q.fc)

        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False  # not update by gradient

        # create the queue and the small queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("small_queue", torch.randn(dim, K))
        self.small_queue = nn.functional.normalize(self.small_queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, small_keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        small_keys = concat_all_gather(small_keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        self.small_queue[:, ptr:ptr + batch_size] = small_keys.transpose(0, 1)

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, img, small_img):
        """
        Input:
            img: a batch of query images
            small_img: a batch of key images: B*6*3*96*96
        Output:
            logit, targets
        """

        # get the shape of small patches
        B, N, C, W, H = small_img.shape

        # compute query features
        q = self.encoder_q(img)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        small_img = small_img.view(-1, C, W, H)
        q_small = self.encoder_q(small_img)  # queries: NxC
        q_small = nn.functional.normalize(q_small, dim=-1)

        # compute key features
        with torch.no_grad():  # no gradient to keys

            k = self.encoder_k(img)  # keys: NxC
            # k = nn.functional.normalize(k, dim=1)

            k_small = self.encoder_k(small_img)  # keys: NxC
            # k_small = nn.functional.normalize(k_small, dim=1)

        # compute the neg-pos logit for larger view
        l_q = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        l_k = torch.einsum('nc,ck->nk', [k, self.queue.clone().detach()])

        l_q_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_k_pos = torch.einsum('nc,nc->n', [k, k]).unsqueeze(-1)

        l_q = torch.cat([l_q_pos, l_q], dim=1)
        l_k = torch.cat([l_k_pos, l_k], dim=1)

        # compute the neg-pos logit for small-views
        l_q_samll = torch.einsum('nc,ck->nk', [q_small, self.small_queue.clone().detach()])
        l_k_samll = torch.einsum('nc,ck->nk', [k_small, self.small_queue.clone().detach()])

        l_q_small_pos = torch.einsum('nc,nc->n', [q_small, k_small]).unsqueeze(-1)
        l_k_small_pos = torch.einsum('nc,nc->n', [k_small, k_small]).unsqueeze(-1)

        l_q_small = torch.cat([l_q_small_pos, l_q_samll], dim=1)
        l_k_small = torch.cat([l_k_small_pos, l_k_samll], dim=1)

        # compute soft labels
        l_q /= self.T
        l_k = nn.functional.softmax(l_k/self.Temp, dim=1)

        l_q_small /= self.T
        l_k_small = nn.functional.softmax(l_k_small/self.Temp, dim=1)

        # labels: positive key indicators for training status monitoring
        labels = torch.zeros(l_q.shape[0], dtype=torch.long).cuda()
        small_labels = torch.zeros(l_k_small.shape[0], dtype=torch.long).cuda()

        # use just one of the small patches as enqueue samples
        k_small = k_small.view(B, N, self.D)

        # de-queue and enqueue
        self._dequeue_and_enqueue(k, k_small[:, 0].contiguous())

        return l_q, l_k, l_q_small, l_k_small, labels, small_labels



# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
