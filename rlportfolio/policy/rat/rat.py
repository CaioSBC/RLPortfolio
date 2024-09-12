import copy
import logging

import torch
from torch import nn
import numpy as np

from rlportfolio.policy.rat.attention import MultiHeadedAttention
from rlportfolio.policy.rat.encoder_decoder import Encoder, Decoder, EncoderDecoder
from rlportfolio.policy.rat.layers import EncoderLayer, DecoderLayer
from rlportfolio.policy.rat.positional_encoding import (
    PositionwiseFeedForward,
    PositionalEncoding,
)


class RAT(EncoderDecoder):
    def __init__(
        self,
        batch_size,
        coin_num,
        window_size,
        feature_number,
        N=6,
        d_model_Encoder=512,
        d_model_Decoder=16,
        d_ff_Encoder=2048,
        d_ff_Decoder=64,
        h=8,
        dropout=0.0,
        local_context_length=3,
        device="cpu",
    ):
        c = copy.deepcopy
        attn_Encoder = MultiHeadedAttention(True, h, d_model_Encoder, 0.1, local_context_length, device)
        attn_Decoder = MultiHeadedAttention(True, h, d_model_Decoder, 0.1, local_context_length, device)
        attn_En_Decoder = MultiHeadedAttention(False, h, d_model_Decoder, 0.1, 1, device)
        ff_Encoder = PositionwiseFeedForward(d_model_Encoder, d_ff_Encoder, dropout)
        ff_Encoder.to(device)
        ff_Decoder = PositionwiseFeedForward(d_model_Decoder, d_ff_Decoder, dropout)
        ff_Decoder.to(device)
        position_Encoder = PositionalEncoding(d_model_Encoder, 0, dropout)
        position_Encoder.to(device)
        position_Decoder = PositionalEncoding(d_model_Decoder, window_size - local_context_length * 2 + 1, dropout)

        super.__init__(batch_size, coin_num, window_size, feature_number, d_model_Encoder, d_model_Decoder,
                        Encoder(EncoderLayer(d_model_Encoder, c(attn_Encoder), c(ff_Encoder), dropout), N),
                        Decoder(DecoderLayer(d_model_Decoder, c(attn_Decoder), c(attn_En_Decoder), c(ff_Decoder),
                                            dropout), N),
                        c(position_Encoder),  # price series position ecoding
                        c(position_Decoder),  # local_price_context position ecoding
                        local_context_length,
                        device)
    
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

        print("Parameters (param name -> param count):")
        for pname, pparams in self.named_parameters():
            pcount = np.prod(pparams.size())
            print(f"\t{pname} -> {pcount}")

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        param_count = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Total param count: {param_count}")

