import torch
from torch import nn
import torch.nn.functional as F

from rlportfolio.utils import create_module_copies
from rlportfolio.policy.rat.layers import LayerNorm


class Encoder(nn.Module):
    "Core encoder is a stack of N layer"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = create_module_copies(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            #            print("Encoder:",x)
            x = layer(x, mask)
        #            print("Encoder:",x.size())
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = create_module_copies(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, price_series_mask, local_price_mask, padding_price):
        for layer in self.layers:
            x = layer(x, memory, price_series_mask, local_price_mask, padding_price)
        return self.norm(x)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(
        self,
        batch_size,
        coin_num,
        window_size,
        feature_number,
        d_model_Encoder,
        d_model_Decoder,
        encoder,
        decoder,
        price_series_pe,
        local_price_pe,
        local_context_length,
        device="cpu",
    ):

        super(EncoderDecoder, self).__init__()
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.batch_size = batch_size
        self.coin_num = coin_num
        self.window_size = window_size
        self.feature_number = feature_number
        self.d_model_Encoder = d_model_Encoder
        self.d_model_Decoder = d_model_Decoder
        self.linear_price_series = nn.Linear(
            in_features=feature_number, out_features=d_model_Encoder
        )
        self.linear_local_price = nn.Linear(
            in_features=feature_number, out_features=d_model_Decoder
        )
        self.price_series_pe = price_series_pe
        self.local_price_pe = local_price_pe
        self.local_context_length = local_context_length
        self.linear_out = nn.Linear(in_features=1 + d_model_Encoder, out_features=1)
        self.linear_out2 = nn.Linear(in_features=1 + d_model_Encoder, out_features=1)
        self.bias = torch.nn.Parameter(torch.zeros([1, 1, 1]))
        self.bias2 = torch.nn.Parameter(torch.zeros([1, 1, 1]))

    def forward(
        self,
        price_series,
        local_price_context,
        previous_w,
        price_series_mask,
        local_price_mask,
        padding_price,
    ):  # [4, 128, 31, 11]
        # price_series:[4,128,31,11]
        price_series = price_series / price_series[0:1, :, -1:, :]
        price_series = price_series.permute(3, 1, 2, 0)  # [4,128,31,11]->[11,128,31,4]
        price_series = price_series.contiguous().view(
            price_series.size()[0] * price_series.size()[1],
            self.window_size,
            self.feature_number,
        )  # [11,128,31,4]->[11*128,31,4]
        price_series = self.linear_price_series(
            price_series
        )  # [11*128,31,3]->[11*128,31,2*12]
        price_series = self.price_series_pe(price_series)  # [11*128,31,2*12]
        price_series = price_series.view(
            self.coin_num, -1, self.window_size, self.d_model_Encoder
        )  # [11*128,31,2*12]->[11,128,31,2*12]
        encode_out = self.encoder(price_series, price_series_mask)
        #        encode_out=self.linear_src_2_embedding(encode_out)
        ###########################padding price#######################################################################################
        if padding_price is not None:
            local_price_context = torch.cat(
                [padding_price, local_price_context], 2
            )  # [11,128,5-1,4] cat [11,128,1,4] -> [11,128,5,4]
            local_price_context = local_price_context.contiguous().view(
                local_price_context.size()[0] * price_series.size()[1],
                self.local_context_length * 2 - 1,
                self.feature_number,
            )  # [11,128,5,4]->[11*128,5,4]
        else:
            local_price_context = local_price_context.contiguous().view(
                local_price_context.size()[0] * price_series.size()[1],
                1,
                self.feature_number,
            )
        ##############Divide by close price################################
        local_price_context = local_price_context / local_price_context[:, -1:, 0:1]
        local_price_context = self.linear_local_price(
            local_price_context
        )  # [11*128,5,4]->[11*128,5,2*12]
        local_price_context = self.local_price_pe(
            local_price_context
        )  # [11*128,5,2*12]
        if padding_price is not None:
            padding_price = local_price_context[
                :, : -self.local_context_length, :
            ]  # [11*128,5-1,2*12]
            padding_price = padding_price.view(
                self.coin_num, -1, self.local_context_length - 1, self.d_model_Decoder
            )  # [11,128,5-1,2*12]
        local_price_context = local_price_context[
            :, -self.local_context_length :, :
        ]  # [11*128,5,2*12]
        local_price_context = local_price_context.view(
            self.coin_num, -1, self.local_context_length, self.d_model_Decoder
        )  # [11,128,5,2*12]
        #################################padding_price=None###########################################################################
        decode_out = self.decoder(
            local_price_context,
            encode_out,
            price_series_mask,
            local_price_mask,
            padding_price,
        )
        decode_out = decode_out.transpose(1, 0)  # [11,128,1,2*12]->#[128,11,1,2*12]
        decode_out = torch.squeeze(decode_out, 2)  # [128,11,1,2*12]->[128,11,2*12]
        previous_w = previous_w.permute(0, 2, 1)  # [128,1,11]->[128,11,1]
        out = torch.cat(
            [decode_out, previous_w], 2
        )  # [128,11,2*12]  cat [128,11,1] -> [128,11,2*12+1]
        ###################################  Decision making ##################################################
        out2 = self.linear_out2(out)  # [128,11,2*12+1]->[128,11,1]
        out = self.linear_out(out)  # [128,11,2*12+1]->[128,11,1]

        bias = self.bias.repeat(out.size()[0], 1, 1)  # [128,1,1]
        bias2 = self.bias2.repeat(out2.size()[0], 1, 1)  # [128,1,1]

        out = torch.cat([bias, out], 1)  # [128,11,1] cat [128,1,1] -> [128,12,1]
        out2 = torch.cat([bias2, out2], 1)  # [128,11,1] cat [128,1,1] -> [128,12,1]

        out = out.permute(0, 2, 1)  # [128,1,12]
        out2 = out2.permute(0, 2, 1)  # [128,1,12]

        out = F.softmax(out, dim=-1)
        out2 = F.softmax(out2, dim=-1)

        out = out * 2
        out2 = -out2
        return out + out2  # [128,1,12]
