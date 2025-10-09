import torch
from torch import nn

from .attention import MultiHeadAttention
from .cache import DecoderCache
from .decoder import Decoder, DecoderLayer
from .embedding import ( LearnedPositionalEmbedding,OutputProjection, PositionalEncoding, TokenEmbedding,)
from .encoder import Encoder, EncoderLayer
from .layers import PositionwiseFeedForward


class Transformer(nn.Module):

    def __init__():
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.encoder = None  
        #build self.encoder and decoder
        self.decoder = None  


    def init_parameters(self, ):

        raise NotImplementedError

    def encode(self,):
        raise NotImplementedError

    def decode():
        raise NotImplementedError

    def forward():
        raise NotImplementedError
