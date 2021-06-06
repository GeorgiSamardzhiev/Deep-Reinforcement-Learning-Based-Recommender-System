import torch
import torch.nn as nn

class InnerProductLayer(nn.Module):
    """InnerProduct Layer used in PNN that compute the element-wise
    product or inner product between feature vectors.
      Input shape
        - a list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.
      Output shape
        - 3D tensor with shape: ``(batch_size, N*(N-1)/2 ,1)`` if use reduce_sum. or 3D tensor with shape:
        ``(batch_size, N*(N-1)/2, embedding_size )`` if not use reduce_sum.
      Arguments
        - **reduce_sum**: bool. Whether return inner product or element-wise product
      References
            - [Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//
            Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 2016: 1149-1154.]
            (https://arxiv.org/pdf/1611.00144.pdf)"""

    def __init__(self, num_inputs, device='cpu'):
        super(InnerProductLayer, self).__init__()
        self.W = nn.Parameter(torch.diag(torch.rand((num_inputs,1))))
        self.W.requires_grad = True
        self.to(device)

    def forward(self, inputs, user):

        embed_list = inputs
        row = []
        col = []
        num_inputs = len(embed_list)

        print('num_inputs', num_inputs)

        embed_list = torch.matmul(self.W, embed_list)
        embed_list = embed_list.unsqueeze(1)

        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                row.append(i)
                col.append(j)

        p = torch.cat([embed_list[idx]
                       for idx in row], dim=1)  # batch num_pairs k
        q = torch.cat([embed_list[idx]
                       for idx in col], dim=1)

        u = user * embed_list

        inner_product = p * q

        u = u.reshape(-1).unsqueeze(0)
        result = torch.cat((u, inner_product), dim=1)
        print('result: ', result)
        return result