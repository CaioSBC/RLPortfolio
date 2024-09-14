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


class RAT(nn.Module):
    def __init__(
        self,
        batch_size=100,
        coin_num=10,
        window_size=50,
        feature_number=3,
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
        super().__init__()
        self.batch_size = batch_size
        self.coin_num = coin_num
        self.window_size = window_size
        self.feature_number = feature_number
        self.local_context_length = local_context_length

        c = copy.deepcopy
        attn_Encoder = MultiHeadedAttention(
            True, h, d_model_Encoder, 0.1, local_context_length, device
        )
        attn_Decoder = MultiHeadedAttention(
            True, h, d_model_Decoder, 0.1, local_context_length, device
        )
        attn_En_Decoder = MultiHeadedAttention(
            False, h, d_model_Decoder, 0.1, 1, device
        )
        ff_Encoder = PositionwiseFeedForward(d_model_Encoder, d_ff_Encoder, dropout)
        ff_Encoder.to(device)
        ff_Decoder = PositionwiseFeedForward(d_model_Decoder, d_ff_Decoder, dropout)
        ff_Decoder.to(device)
        position_Encoder = PositionalEncoding(d_model_Encoder, 0, dropout)
        position_Encoder.to(device)
        position_Decoder = PositionalEncoding(
            d_model_Decoder, window_size - local_context_length * 2 + 1, dropout
        )

        self.encoder_decoder = EncoderDecoder(
            batch_size,
            coin_num,
            window_size,
            feature_number,
            d_model_Encoder,
            d_model_Decoder,
            Encoder(
                EncoderLayer(d_model_Encoder, c(attn_Encoder), c(ff_Encoder), dropout),
                N,
            ),
            Decoder(
                DecoderLayer(
                    d_model_Decoder,
                    c(attn_Decoder),
                    c(attn_En_Decoder),
                    c(ff_Decoder),
                    dropout,
                ),
                N,
            ),
            c(position_Encoder),  # price series position ecoding
            c(position_Decoder),  # local_price_context position ecoding
            local_context_length,
            device,
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        print("Parameters (param name -> param count):")
        for pname, pparams in self.named_parameters():
            pcount = np.prod(pparams.size())
            print(f"\t{pname} -> {pcount}")

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        param_count = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Total param count: {param_count}")

    def forward(
        self, observation: torch.Tensor, last_action: torch.Tensor
    ) -> torch.Tensor:
        last_action = self._process_last_action(last_action)
        observation = observation.permute(
            1, 0, 3, 2
        )  # shape (features, batch, time_window, num_stocks)

        # taken from original code
        price_series_mask = (torch.ones(observation.size()[1], 1, self.window_size) == 1) # [128, 1, 31]
        currt_price = observation.permute((3, 1, 2, 0))  # [4,128,31,11]->[11,128,31,4]
        if self.local_context_length > 1:
            padding_price = currt_price[:, :, -self.local_context_length * 2 + 1:-1, :]
        else:
            padding_price = None
        currt_price = currt_price[:, :, -1:, :]  # [11,128,31,4]->[11,128,1,4]
        trg_mask = self._make_std_mask(currt_price, observation.size()[1])

        output = self.encoder_decoder.forward(
            observation, 
            currt_price,
            last_action,
            price_series_mask,
            trg_mask,
            padding_price,
        ) # [128,1,12]

        output = torch.squeeze(output) # [128, 12]
        return output

    def _process_last_action(
        self, last_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process the last action to retrieve cash bias and last stocks.

        Args:
          last_action: Last performed action.

        Returns:
            Reshaped last action performed.
        """
        batch_size = last_action.shape[0]
        stocks = last_action.shape[1]

        return last_action.reshape((batch_size, stocks, 1))
    

    def _subsequent_mask(self, size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0


    def _make_std_mask(self, local_price_context, batch_size):
        "Create a mask to hide padding and future words."
        local_price_mask = (torch.ones(batch_size, 1, 1) == 1)
        local_price_mask = local_price_mask & (self._subsequent_mask(local_price_context.size(-2)).type_as(local_price_mask.data))
        return local_price_mask
