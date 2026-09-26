import math
import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from math import ceil as up

def _weights_init(m):
    classname = m.__class__.__name__

    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class Aux_net(nn.Module):
    def __init__(self, dim,  num_classes=10):
        super(Aux_net, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, num_classes)

        self.apply(_weights_init)
    def forward(self, x, ext_mask=None):
        out = F.avg_pool2d(x, x.size()[3])
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas

class Concat_models(nn.Module):
    def __init__(self, model_1, model_2):
        super(Concat_models, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
    def forward(self,x):
        rst_tmp = self.model_1(x)
        rst = self.model_2(rst_tmp)
        return rst_tmp, rst

class Aux_net_v2(nn.Module):
    def __init__(self, dim,  num_classes=10):
        super(Aux_net_v2, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, int(dim/2))
        self.bn1 = nn.BatchNorm1d(num_features=int(dim/2))
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(int(dim/2), num_classes)

        self.apply(_weights_init)
    def forward(self, x, ext_mask=None):
        out = F.avg_pool2d(x, x.size()[3])
        out = out.view(out.size(0), -1)
        out = self.relu(self.bn1(self.linear(out)))
        logits = self.linear2(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas


class Aux_net_RoBerta(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2):
        super(Aux_net_RoBerta, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size // 2)
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(hidden_size // 2, num_classes)
        self.apply(_weights_init)

    def forward(self, x, ext_mask=None):
        out = x[:, 0, :]
        out = self.gelu(self.bn1(self.linear(out)))
        logits = self.linear2(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas


class Aux_net_RoBerta_l(nn.Module):
    def __init__(self, hidden_size=1024, num_classes=2):
        super(Aux_net_RoBerta_l, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size // 2)
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(hidden_size // 2, num_classes)
        self.apply(_weights_init)

    def forward(self, x, ext_mask=None):
        out = x[:, 0, :]
        out = self.gelu(self.bn1(self.linear(out)))
        logits = self.linear2(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas


class Aux_net_ViT(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(Aux_net_ViT, self).__init__()
        self.fc = nn.Linear(embed_dim, num_classes)
        self.apply(_weights_init)

    def forward(self, x, ext_mask=None):
        cls    = x[:, 0, :]
        logits = self.fc(cls)
        probas = F.softmax(logits, dim=1)
        return logits, probas


class Concat_models(nn.Module):
    def __init__(self, model_1, model_2):
        super(Concat_models, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
    def forward(self,x):
        rst_tmp = self.model_1(x)
        rst = self.model_2(rst_tmp)
        return rst_tmp, rst

class Aux_net_ratio(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2, hidden=2):
        super(Aux_net_ratio, self).__init__()
        h = max(1, int(hidden))
        self.linear  = nn.Linear(hidden_size, h)
        self.bn1     = nn.BatchNorm1d(h)
        self.gelu    = nn.GELU()
        self.linear2 = nn.Linear(h, num_classes)
        self.apply(_weights_init)

    def forward(self, x, ext_mask=None):
        out = x[:, 0, :]
        out = self.gelu(self.bn1(self.linear(out)))
        logits = self.linear2(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas


def aux_hidden_from_ratio(n_trainable_total, in_dim, num_classes, ratio, h_min=0):
    target = float(ratio) * float(n_trainable_total)
    h = int(round((target - num_classes) / float(in_dim + 3 + num_classes)))
    h = max(1, h)
    if int(h_min) > 0:
        h = max(h, int(h_min))
    return h


class Aux_net_ffn(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2, hidden=8, p_drop=0.1):
        super(Aux_net_ffn, self).__init__()
        h = max(1, int(hidden))
        self.ln    = nn.LayerNorm(hidden_size)
        self.lin1  = nn.Linear(hidden_size, h)
        self.gelu  = nn.GELU()
        self.lin2  = nn.Linear(h, hidden_size)
        self.drop  = nn.Dropout(float(p_drop))
        self.head  = nn.Linear(hidden_size, num_classes)
        self.apply(_weights_init)

    def forward(self, x, ext_mask=None):
        z = x[:, 0, :]
        y = z + self.drop(self.lin2(self.gelu(self.lin1(self.ln(z)))))
        logits = self.head(y)
        probas = F.softmax(logits, dim=1)
        return logits, probas


def aux_hidden_ffn_from_ratio(n_trainable_total, in_dim, num_classes, ratio, h_min=0):
    d = int(in_dim); C = int(num_classes)
    fixed = 3 * d + d * C + C
    h = int(round((float(ratio) * float(n_trainable_total) - fixed) / float(2 * d + 1)))
    h = max(1, h)
    if int(h_min) > 0:
        h = max(h, int(h_min))
    return h


class Aux_net_head(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2, readout=None, readout_kind='pooler'):
        super(Aux_net_head, self).__init__()
        self.readout_kind = str(readout_kind)
        self.readout = readout
        self.fc = nn.Linear(hidden_size, num_classes)
        init.kaiming_normal_(self.fc.weight)
        if self.readout is not None:
            for p in self.readout.parameters():
                p.requires_grad_(False)

    def forward(self, x, ext_mask=None):
        if self.readout_kind == 'pooler':
            out = self.readout(x)
        elif self.readout_kind == 'cls_mlp':
            out = F.relu(self.readout(x[:, 0, :]))
        else:
            out = x[:, 0, :]
        logits = self.fc(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas

    def train(self, mode=True):
        super(Aux_net_head, self).train(mode)
        if self.readout is not None:
            self.readout.eval()
        return self


class Aux_net_reg(nn.Module):

    def __init__(self, in_dim, num_classes=1, pool='auto'):
        super(Aux_net_reg, self).__init__()
        self.pool = pool
        self.fc = nn.Linear(in_dim, 1)
        self.apply(_weights_init)

    def forward(self, x, ext_mask=None):
        if x.dim() == 4:
            z = x.mean(dim=(1, 2))
        elif x.dim() == 3:
            z = x.mean(dim=1) if self.pool == 'mean' else x[:, 0, :]
        elif x.dim() == 2:
            z = x
        else:
            raise RuntimeError(f"[AUX-REG] {tuple(x.shape)}")
        logits = self.fc(z)
        return logits, logits


class Aux_net_qa(nn.Module):

    def __init__(self, in_dim, num_classes=2, pool='auto'):
        super(Aux_net_qa, self).__init__()
        self.fc = nn.Linear(in_dim, 2)
        self.apply(_weights_init)

    def forward(self, x, ext_mask=None):
        if x.dim() != 3:
            raise RuntimeError(f"[AUX-QA] {tuple(x.shape)}")
        se = self.fc(x)
        start, end = se[..., 0], se[..., 1]
        if ext_mask is not None:
            m = ext_mask

            m_add = (m[:, 0, -1, :].to(start.dtype) if m.dim() == 4
                     else (1.0 - m.to(start.dtype)) * -1e4)
            start = start + m_add
            end = end + m_add
        logits = torch.cat([start, end], dim=0)
        return logits, F.softmax(logits, dim=1)


class Aux_net_fc(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2, pool='auto'):
        super(Aux_net_fc, self).__init__()
        assert pool in ('auto', 'cls', 'mean'), f"[AUX-POOL] pool={pool}"
        self.pool = pool
        self.hidden_size = int(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.apply(_weights_init)

    def _pool(self, x):

        if x.dim() == 4:
            z = x.mean(dim=(1, 2)) if self.pool in ('auto', 'mean') else x[:, 0, 0, :]
        elif x.dim() == 3:
            z = x.mean(dim=1) if self.pool == 'mean' else x[:, 0, :]
        elif x.dim() == 2:
            z = x
        else:
            raise RuntimeError(f"[AUX-POOL] {tuple(x.shape)}")
        if z.size(-1) != self.fc.in_features:
            raise RuntimeError(
                f"[AUX-POOL] {z.size(-1)} vs "
                f"Linear.in_features {self.fc.in_features}. "
                f"model_assign _in '{tuple(x.shape)}' ")
        return z

    def forward(self, x, ext_mask=None):
        z = self._pool(x)
        logits = self.fc(z)
        return logits, F.softmax(logits, dim=1)


class Aux_net_headh(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2, hidden=2, p_drop=0.1):
        super(Aux_net_headh, self).__init__()
        self.h = int(hidden)
        self.dropout = nn.Dropout(float(p_drop))
        if self.h > 0:
            self.dense    = nn.Linear(hidden_size, self.h)
            self.out_proj = nn.Linear(self.h, num_classes)
        else:
            self.dense    = None
            self.out_proj = nn.Linear(hidden_size, num_classes)
        self.apply(_weights_init)

    def forward(self, x, ext_mask=None):
        z = x[:, 0, :]
        z = self.dropout(z)
        if self.dense is not None:
            z = self.dense(z)
            z = torch.tanh(z)
            z = self.dropout(z)
        logits = self.out_proj(z)
        return logits, F.softmax(logits, dim=1)


def aux_hidden_headh(n_total_trainable, in_dim, num_classes, ratio, h_min=0, h_fixed=0):
    if int(h_fixed) > 0:
        return int(h_fixed)
    d, C = int(in_dim), int(num_classes)
    h = int(round((float(ratio) * float(n_total_trainable) - C) / float(d + 1 + C)))
    h = max(0, h)
    return max(h, int(h_min)) if int(h_min) > 0 else h


def aux_hidden_from_scale(in_dim, num_classes, scale, h_min=0):
    d, C = int(in_dim), int(num_classes)
    base = d * C + C
    target = float(scale) * base
    h = int(round((target - C) / float(d + 1 + C)))
    h = max(1, h)
    if int(h_min) > 0:
        h = max(h, int(h_min))
    return h


class Aux_net_fc_scaled(nn.Module):

    def __init__(self, hidden_size=768, num_classes=2, hidden=8, pool='auto'):
        super(Aux_net_fc_scaled, self).__init__()
        assert pool in ('auto', 'cls', 'mean'), f"[AUX-POOL] pool={pool}"
        self.pool = pool
        self.hidden_size = int(hidden_size)
        h = max(1, int(hidden))
        self.fc1 = nn.Linear(hidden_size, h)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(h, num_classes)
        self.apply(_weights_init)

    def forward(self, x, ext_mask=None):

        if x.dim() == 4:
            z = x.mean(dim=(1, 2)) if self.pool in ('auto', 'mean') else x[:, 0, 0, :]
        elif x.dim() == 3:
            z = x.mean(dim=1) if self.pool == 'mean' else x[:, 0, :]
        elif x.dim() == 2:
            z = x
        else:
            raise RuntimeError(f"[AUX-POOL] {tuple(x.shape)}")
        if z.size(-1) != self.fc1.in_features:
            raise RuntimeError(
                f"[AUX-POOL] {z.size(-1)} vs "
                f"Linear.in_features {self.fc1.in_features}")
        logits = self.fc2(self.act(self.fc1(z)))
        return logits, F.softmax(logits, dim=1)


class Aux_net_lm(nn.Module):
    def __init__(self, in_dim, vocab, pool='auto'):
        super(Aux_net_lm, self).__init__()
        self.fc = nn.Linear(in_dim, int(vocab))
        self.apply(_weights_init)

    def forward(self, x, ext_mask=None):
        if x.dim() != 3:
            raise RuntimeError(f"[AUX-LM] {tuple(x.shape)}")
        logits = self.fc(x)[:, :-1, :].contiguous()
        logits = logits.reshape(-1, logits.size(-1))
        return logits, logits

    @torch.no_grad()
    def step_logits(self, x, ext_mask=None):
        return self.fc(x)

    @torch.no_grad()
    def pos_logits(self, x, ext_mask, pos):
        return self.fc(x[torch.arange(x.size(0), device=x.device), pos])


class Aux_net_mlp1(Aux_net_fc):
    def __init__(self, hidden_size=768, num_classes=2, pool='auto', h=None):
        super().__init__(hidden_size, num_classes, pool)
        h = int(h) if h else max(1, int(hidden_size) // 2)
        self.fc = nn.Linear(hidden_size, h); self.act = nn.GELU(); self.fc2 = nn.Linear(h, num_classes)
        self.apply(_weights_init)

    def forward(self, x, ext_mask=None):
        z = self._pool(x)
        logits = self.fc2(self.act(self.fc(z)))
        return logits, F.softmax(logits, dim=1)


class Aux_net_mlpN(Aux_net_fc):
    def __init__(self, hidden_size=768, num_classes=2, pool='auto', h=None, depth=4):
        super().__init__(hidden_size, num_classes, pool)
        h = int(h) if h else max(1, int(hidden_size) // 2); depth = max(1, int(depth))
        layers = [nn.Linear(hidden_size, h), nn.GELU()]
        for _ in range(depth - 1): layers += [nn.Linear(h, h), nn.GELU()]
        self.mlp = nn.Sequential(*layers); self.fc2 = nn.Linear(h, num_classes); self.depth = depth
        self.fc = layers[0]
        self.apply(_weights_init)

    def forward(self, x, ext_mask=None):
        z = self._pool(x)
        logits = self.fc2(self.mlp(z))
        return logits, F.softmax(logits, dim=1)


class Aux_net_fc_lora(Aux_net_fc):
    def __init__(self, hidden_size=768, num_classes=2, pool='auto', r=4, alpha=8, p_drop=0.1):
        super().__init__(hidden_size, num_classes, pool)
        self.fc.weight.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(int(r), int(hidden_size))); init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.lora_B = nn.Parameter(torch.zeros(int(num_classes), int(r)))
        self.scaling = float(alpha) / float(r); self.lora_drop = nn.Dropout(float(p_drop))

    def forward(self, x, ext_mask=None):
        z = self._pool(x)
        logits = self.fc(z) + (self.lora_drop(z) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return logits, F.softmax(logits, dim=1)
