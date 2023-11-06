import torch
import torch.nn as nn

class AlignedVAMemory(nn.Module):
    def __init__(self, bank_size=32):
        super(AlignedVAMemory, self).__init__()
        self.n_mu = bank_size
        self.n_class = 28
        self.out_dim = 256

        self.register_buffer("cls_v_queue", torch.zeros(self.n_class, self.n_mu, 7, 7, 512))
        self.register_buffer("cls_a_queue", torch.zeros(self.n_class, self.n_mu, 128))
        self.register_buffer("cls_sc_queue", torch.zeros(self.n_class, self.n_mu))

    @torch.no_grad()
    def _update_queue(self, inp_v, inp_a, inp_sc, cls_idx):
        # for idx in cls_idx:
        #     self._sort_permutation(inp_mu, inp_sc, idx)
        for i in range(len(cls_idx)):
            self._sort_permutation(inp_v[i], inp_a[i], inp_sc[i], cls_idx[i])


    @torch.no_grad()
    def _sort_permutation(self, inp_v, inp_a, inp_sc, idx):
        sf = self.cls_a_queue[idx].sum(-1)
        sa = inp_a.sum(-1)
        if sa not in sf:
            concat_sc = torch.cat([self.cls_sc_queue[idx], inp_sc.unsqueeze(0)], 0)
            concat_v = torch.cat([self.cls_v_queue[idx], inp_v.unsqueeze(0)], 0)
            concat_a = torch.cat([self.cls_a_queue[idx], inp_a.unsqueeze(0)], 0)
            sorted_sc, indices = torch.sort(concat_sc, descending=True)
            sorted_v = torch.index_select(concat_v, 0, indices[:self.n_mu])
            sorted_a = torch.index_select(concat_a, 0, indices[:self.n_mu])
            self.cls_v_queue[idx] = sorted_v
            self.cls_a_queue[idx] = sorted_a
            self.cls_sc_queue[idx] = sorted_sc[:self.n_mu]

