import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformer_decoder import TransformerDecoderLayer
from memory_bank import AlignedVAMemory


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(dim=1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(dim=0).transpose(0, 1)  # shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class FusionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(FusionLayer, self).__init__()
        self.hidden_dim = hidden_dim

        # self.ln1 = nn.LayerNorm(hidden_dim)
        self.layers = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, feats1, feats2):
        feats = torch.cat([feats1, feats2], dim=-1)
        # feats = self.ln1(feats)
        feats = self.layers(feats.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        feats = self.ln2(feats)
        return feats


class BGClassifier(nn.Module):
    def __init__(self, hidden_dim):
        super(BGClassifier, self).__init__()

        self.linear2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, feats):
        preds = self.linear2(feats).squeeze(-1)
        preds = self.dropout(preds)
        preds = torch.sigmoid(preds)
        return preds


class EVNClassifier(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(EVNClassifier, self).__init__()

        self.linear2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, feats):
        feats_cls = self.linear2(feats)
        feats_cls = feats_cls.mean(1)
        preds = self.dropout(feats_cls)
        return preds


class MEME_WS(nn.Module):
    def __init__(self, a_dim=128, v_dim=512, hidden_dim=256, category_num=29, num_layers=2, bank_size=16, margin=0.1):
        super(MEME_WS, self).__init__()

        self.hidden_dim = hidden_dim
        self.a_dim = a_dim
        self.v_dim = v_dim
        self.num_layers = num_layers
        self.margin = margin

        self.audio_fc = nn.Sequential(nn.Linear(a_dim, hidden_dim),
                                      nn.LeakyReLU(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Dropout(0.2))

        self.video_fc = nn.Sequential(nn.Linear(v_dim, hidden_dim),
                                      nn.LeakyReLU(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Dropout(0.2))

        self.vstf_pe = PositionalEncoding(self.hidden_dim)
        video_stf = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=1024)
        self.video_stf = nn.TransformerEncoder(video_stf, num_layers=self.num_layers)

        spatial_decoder = TransformerDecoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=1024)
        self.spatial_decoder = nn.TransformerDecoder(spatial_decoder, num_layers=self.num_layers)

        audio_tf = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=1024)
        self.audio_tf = nn.TransformerEncoder(audio_tf, num_layers=self.num_layers)

        video_ttf = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=1024)
        self.video_ttf = nn.TransformerEncoder(video_ttf, num_layers=self.num_layers)

        temporal_decoder = TransformerDecoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=1024)
        self.temporal_decoder = nn.TransformerDecoder(temporal_decoder, num_layers=self.num_layers)

        self.fusion_layer = FusionLayer(hidden_dim)

        self.audio_classifier = nn.Linear(hidden_dim, category_num - 1)
        self.video_classifier = nn.Linear(hidden_dim, category_num - 1)

        self.kl_loss = nn.KLDivLoss(reduction='none')
        self.aligned_bank = AlignedVAMemory(bank_size=bank_size)

        self.final_event_classifier = EVNClassifier(hidden_dim, category_num - 1)
        self.final_bg_classifier = BGClassifier(hidden_dim)

    def fusion(self, audio, video):
        # shape of audio: [bs, 10, 128]
        # shape of video: [bs, 10, 7, 7, 512]
        B, T, H, W, C = video.shape

        afeats = self.audio_fc(audio)  # shape: bs * 10 * hidden_dim
        vfeats_full = self.video_fc(video).permute(0, 1, 4, 2, 3)
        vfeats_full = vfeats_full.view(-1, *vfeats_full.shape[2:])

        v_spat = vfeats_full.view(*vfeats_full.shape[0:2], -1).permute(2, 0, 1)  # shape: 49 * (bs*10) * dim
        v_spat = self.vstf_pe(v_spat)
        v_spat = self.video_stf(v_spat)
        a_spat = afeats.reshape(-1, self.hidden_dim).unsqueeze(0)  # shape: 1 * (bs*10) * dim
        av_feats1 = self.spatial_decoder(a_spat, v_spat).reshape(B, -1, self.hidden_dim)  # audio feats that consider video feats

        afeats = self.audio_tf(afeats.permute(1, 0, 2)).permute(1, 0, 2)
        vfeats_full = v_spat.reshape(7, 7, -1, self.hidden_dim).permute(2, 3, 0, 1)
        vfeats = vfeats_full.mean(-1).mean(-1)
        vfeats = vfeats.reshape(B, -1, self.hidden_dim)
        vfeats = self.video_ttf(vfeats.permute(1, 0, 2)).permute(1, 0, 2)
        av_feats2 = self.temporal_decoder(afeats.permute(1, 0, 2), vfeats.permute(1, 0, 2)).permute(1, 0, 2)  # video feats that consider audio feats

        fused_feats = self.fusion_layer(av_feats1, av_feats2)

        return fused_feats, afeats, vfeats

    def forward(self, audio, video, one_hot_labels=None):
        # shape of audio: [bs, 10, 128]
        # shape of video: [bs, 10, 7, 7, 512]
        B, T, H, W, C = video.shape

        fused_feats, afeats, vfeats = self.fusion(audio, video)

        audio_evn_preds = self.audio_classifier(afeats)
        video_evn_preds = self.video_classifier(vfeats)

        final_bg_preds = self.final_bg_classifier(fused_feats)
        final_evn_pred = self.final_event_classifier(fused_feats)

        if one_hot_labels is not None:
            # generating event labels
            labels_foreground = one_hot_labels[:, :-1]  # shape: [bs, 28]
            _, labels_event = labels_foreground.max(-1)

            audio_dist = F.softmax(audio_evn_preds.reshape(B * T, -1), dim=1)
            video_dist = F.softmax(video_evn_preds.reshape(B * T, -1), dim=1)
            mean_dist = (audio_dist + video_dist) / 2
            jsd = (self.kl_loss(mean_dist.log(), audio_dist) +
                   self.kl_loss(mean_dist.log(), video_dist)) / 2
            jsd = jsd.sum(1).reshape(B, T)
            relation_scores = 1 - jsd
            selected_scores, selected_idx = relation_scores.max(dim=1)
            selected_idx = F.one_hot(selected_idx, num_classes=10).unsqueeze(-1)
            selected_audio = (audio * selected_idx).sum(1)
            selected_video = (video * selected_idx.unsqueeze(-1).unsqueeze(-1)).sum(1)
            self.aligned_bank._update_queue(selected_video, selected_audio, selected_scores, labels_event)

            vq = self.aligned_bank.cls_v_queue
            aq = self.aligned_bank.cls_a_queue
            aligned_memory, _, _ = self.fusion(aq, vq)  # shape: num_classes * len_queue * hidden_dim

            pos_mem = aligned_memory[labels_event]  # shape: B * len_queue * hidden_dim

            prototypes = aligned_memory.mean(dim=1)  # shape: num_classes * hidden_dim

            pdists = (torch.cdist(fused_feats, prototypes) / math.sqrt(self.hidden_dim)).mean(1)
            rf_evn_pred = - pdists / 0.1

            dists = torch.cdist(fused_feats, pos_mem) / math.sqrt(self.hidden_dim)
            max_dist, _ = dists.max(-1)
            min_dist, _ = dists.min(-1)

            rf_bg_loss = final_bg_preds * max_dist + (1 - final_bg_preds) * F.relu(self.margin - min_dist)

            return final_bg_preds, final_evn_pred, audio_evn_preds, video_evn_preds, rf_bg_loss, rf_evn_pred
        else:

            vq = self.aligned_bank.cls_v_queue
            aq = self.aligned_bank.cls_a_queue
            aligned_memory, _, _ = self.fusion(aq, vq)  # shape: num_classes * len_queue * hidden_dim
            prototypes = aligned_memory.mean(dim=1)  # shape: num_classes * hidden_dim

            pdists = (torch.cdist(fused_feats, prototypes) / math.sqrt(self.hidden_dim)).mean(1)
            rf_evn_pred = - pdists / 0.1

            final_evn_pred = F.softmax(rf_evn_pred, dim=-1) + F.softmax(final_evn_pred, dim=-1)

            pred_labels_event = final_evn_pred.max(dim=1)[1]
            pos_mem = aligned_memory[pred_labels_event]  # shape: B * len_queue * hidden_dim

            dists = torch.cdist(fused_feats, pos_mem) / math.sqrt(self.hidden_dim)
            min_dist, _ = dists.min(-1)

            final_bg_preds = F.sigmoid(self.margin - min_dist)

            return final_bg_preds, final_evn_pred
