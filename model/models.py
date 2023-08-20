""" Implementation of DeepConvLSTM
Reference:
- https://www.mdpi.com/1424-8220/16/1/115
"""
import torch
from torch import nn
from openpack_toolkit import OPENPACK_OPERATIONS
import math

class DeepConvLSTM(nn.Module):
    """Imprementation of DeepConvLSTM [Sensors 2016].

    Note:
        https://www.mdpi.com/1424-8220/16/1/115 (Sensors, 2016)

    """

    def __init__(self, in_ch: int = 6, num_classes: int = None):
        super().__init__()

        # NOTE: The first block is input layer.

        # -- [L2-5] Convolutions --
        blocks = []
        for i in range(4):
            in_ch_ = in_ch if i == 0 else 64
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch_, 64, kernel_size=(5, 1), padding=(2, 0)),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                )
            )
        self.conv2to5 = nn.ModuleList(blocks)

        # -- [L6-7] LSTM --
        hidden_units = 128
        self.lstm6 = nn.LSTM(64, hidden_units, batch_first=True)
        self.lstm7 = nn.LSTM(hidden_units, hidden_units, batch_first=True)
        self.dropout6 = nn.Dropout(p=0.5)
        self.dropout7 = nn.Dropout(p=0.5)

        # -- [L8] Softmax Layer (Output Layer) --
        self.out8 = nn.Conv2d(
            hidden_units,
            num_classes,
            1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape = (B, CH, T, 1)
        Returns:
            torch.Tensor: shape = (B, N_CLASSES, T, 1)
        """
        # -- Conv --
        for i in range(4):
            x = self.conv2to5[i](x)

        # -- LSTM --
        # Reshape: (B, CH, 1, T) -> (B, T, CH)
        x = x.squeeze(3).transpose(1, 2)

        x, _ = self.lstm6(x)
        x = self.dropout6(x)
        x, _ = self.lstm7(x)
        x = self.dropout7(x)

        # -- [L8] Softmax Layer (Output Layer) --
        # Reshape: (B, T, CH) -> (B, CH, T, 1)
        x = x.transpose(1, 2).unsqueeze(3)
        x = self.out8(x)
        return x


class DeepConvLSTMSelfAttn(nn.Module):
    """Imprementation of a DeepConvLSTM with Self-Attention used in ''Deep ConvLSTM with
    self-attention for human activity decoding using wearable sensors'' (Sensors 2020).

    Note:
        https://ieeexplore.ieee.org/document/9296308 (Sensors 2020)
    """

    def __init__(
        self,
        in_ch: int = 46,
        num_classes: int = None,
        cnn_filters=3,
        lstm_units=32,
        num_attn_heads: int = 2,
    ):
        super().__init__()
        if num_classes is None:
            num_classes = len(OPENPACK_OPERATIONS)

        # NOTE: The first block is input layer.

        # -- [1] Embedding Layer --
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, cnn_filters, kernel_size=1, padding=0),
            nn.BatchNorm2d(cnn_filters),
            nn.ReLU(),
        )

        # -- [2] LSTM Encoder --
        self.lstm = nn.LSTM(cnn_filters, lstm_units, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)

        # -- [3] Self-Attention --
        self.attention = nn.MultiheadAttention(
            lstm_units,
            num_attn_heads,
            batch_first=True,
        )

        # -- [4] Softmax Layer (Output Layer) --
        self.out = nn.Conv2d(
            lstm_units,
            num_classes,
            1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape = (B, CH, T, 1)
        Returns:
            torch.Tensor: shape = (B, N_CLASSES, T, 1)
        """
        # -- [1] Embedding Layer --
        x = self.conv(x)

        # -- [2] LSTM Encoder --
        # Reshape: (B, CH, 1, T) -> (B, T, CH)
        x = x.squeeze(3).transpose(1, 2)

        x, _ = self.lstm(x)
        x = self.dropout(x)

        # -- [3] Self-Attention --
        x, w = self.attention(x.clone(), x.clone(), x.clone())

        # -- [4] Softmax Layer (Output Layer) --
        # Reshape: (B, T, CH) -> (B, CH, T, 1)
        x = x.transpose(1, 2).unsqueeze(3)
        x = self.out(x)
        return x
    
class DeepConvLstmV3(nn.Module):
    def __init__(self, in_ch: int = 46, num_classes: int = None):
        super().__init__()
        if num_classes is None:
            num_classes = len(OPENPACK_OPERATIONS)

        # -- [1] CNN --
        # *** Edit Here ***
        num_conv_layers = 4 # convolutional layers (Default: 4)
        num_conv_filter = 64 # convolutional filters (Default: 64)
        ks = 5 # kernel size, 
        # ******************
        
        blocks = []
        for i in range(num_conv_layers):
            in_ch_ = in_ch if i == 0 else 64
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch_, 64, kernel_size=(5, 1), padding=(2, 0)),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                )
            )
        self.conv_blocks = nn.ModuleList(blocks)

        # -- [2] LSTM --
        # *** Edit Here ***        
        hidden_units = 128 # number of hidden units for Bi-LSTM
        # ******************
        
        # NOTE: enable ``bidirectional``
        self.lstm6 = nn.LSTM(num_conv_filter, hidden_units, batch_first=True, bidirectional=True)
        self.lstm7 = nn.LSTM(hidden_units*2, hidden_units, batch_first=True,  bidirectional=True)
        self.dropout6 = nn.Dropout(p=0.5)
        self.dropout7 = nn.Dropout(p=0.5)

        # -- [3] Output --
        self.out8 = nn.Conv2d(
            hidden_units * 2,
            num_classes,
            1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape = (B, CH, T, 1)
        Returns:
            torch.Tensor: shape = (B, N_CLASSES, T, 1)
        """
        # -- [1] Conv --
        for block in self.conv_blocks:
            x = block(x)

        # -- [2] LSTM --
        # Reshape: (B, CH, 1, T) -> (B, T, CH)
        x = x.squeeze(3).transpose(1, 2)

        x, _ = self.lstm6(x)
        x = self.dropout6(x)
        x, _ = self.lstm7(x)
        x = self.dropout7(x)

        # Reshape: (B, T, CH) -> (B, CH, T, 1)
        x = x.transpose(1, 2).unsqueeze(3) 
        
        # -- [3] Output --
        x = self.out8(x)
        return x

class CSNetWithFusion(nn.Module):
    def __init__(self, in_ch: int = 46, num_classes: int = None):
        super().__init__()
        if num_classes is None:
            num_classes = len(OPENPACK_OPERATIONS)
        
        self.imu_block = CSNetBlock(in_ch=12,num_classes=num_classes, reshape_len=225)
        self.keypoints_block = CSNetBlock(in_ch=34, num_classes=num_classes, reshape_len=113)
        self.e4_block =  CSNetBlock(in_ch=6, num_classes=num_classes, reshape_len=240)
        self.out = nn.Conv1d(
            3,
            num_classes,
            3,
            stride=1,
            padding="same",
        )
        

    def forward(self, data) -> torch.Tensor: 
       imu = data[0]
       keypoints = data[1]
       e4 = data[2]

       imu_x = self.imu_block(imu)
       keypoints_x = self.keypoints_block(keypoints)
       e4_x = self.e4_block(e4)

       #print(f"imu shape {imu_x.shape} keypoints shape {keypoints_x.shape} e4 shape {e4_x.shape} ")

       #x = torch.cat([imu_x, keypoints_x, e4_x],dim=2)
       x = torch.stack([imu_x, keypoints_x, e4_x], dim=0)
       x = x.permute(1,0,2)
       x = self.out(x)
       return x
    

class CSNetBlock(nn.Module):
    def __init__(self, in_ch: int = 46, num_classes: int = None, reshape_len = 0):
        super().__init__()
        if num_classes is None:
            num_classes = len(OPENPACK_OPERATIONS)
        
        self.conv1 = ConvolutionBlock(in_ch)        
        self.pos = PositionalEncoding(32)
        self.attn1 = SelfAttentionBlock()
        self.attn2 = SelfAttentionBlock()
        self.conv2 = ConvolutionBlock(in_ch=32)
        self.maxpool = nn.MaxPool1d(kernel_size=1, stride=2)
        self.reshape = ReshapeBlock(length=reshape_len)
        """ self.out = nn.Conv1d(
            64,
            num_classes,
            1,
            stride=1,
            padding=0,
        ) """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        x = self.conv1(x)
        #print("x after conv shape ", x.shape)
        # Reshape: (B, CH, T) -> (T, B, CH)
        x = x.permute(2, 0, 1)
        x = self.pos(x)
        # Reshape: (T, B, CH) -> (B, T, CH)
        x = x.permute(1,0,2)
        #print("x after perm ", x.shape)
        x = self.attn1(x)
        x = self.attn2(x)
        # Reshape: (B, T, CH) -> (B, CH, T)
        x = x.permute(0,2,1)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.reshape(x)        
        #print("x after conv2", x.shape)
        #x = self.out(x)
        return x
    
class ConvolutionBlock(nn.Module):
    def __init__(self, in_ch: int = 1, num_layers: int = 2, k: int = 3, filters = [32,32]):
        super().__init__()        

        blocks = []
        for i in range(num_layers):
            in_ch_ = in_ch if i == 0 else filters[i-1]
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_ch_, filters[i], kernel_size=k, padding='same'),
                    nn.BatchNorm1d(filters[i]),
                    nn.ReLU(),
                )
            )
        self.conv_sub_blocks = nn.ModuleList(blocks)
        self.max_pool = nn.MaxPool1d(kernel_size=k, stride=2,padding=1)
       

    def forward(self, x: torch.Tensor) -> torch.Tensor:         
        for sub_block in self.conv_sub_blocks:
            #print("iter")
            x = sub_block(x)
        #print("x shape", x.shape)
        x = self.max_pool(x)
        return x

class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_size=32, heads=8, dropout=0.1):
        super(SelfAttentionBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(embed_size)
        self.multihead_attention = nn.MultiheadAttention(embed_size, heads)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size)
        )
    
    def forward(self, x):
        residual = x
        
        x = self.norm1(x)
        attention_output, _ = self.multihead_attention(x, x, x)
        x = residual + self.dropout(attention_output)
        
        residual = x
        
        x = self.norm2(x)
        feed_forward_output = self.feed_forward(x)
        x = residual + self.dropout(feed_forward_output)
        
        return x
    
class PositionalEncoding(nn.Module):
#https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class FusionResultsWithConv(nn.Module):
    def __init__(self, input_shapes, seq_length = 1800, num_classes = 11):
        super().__init__()
        self.linear1 = nn.Linear(input_shapes[0][-1], seq_length)
        self.linear2 = nn.Linear(input_shapes[1][-1], seq_length)
        self.linear3 = nn.Linear(input_shapes[2][-1], seq_length)
        
    def forward(self, tensor1, tensor2, tensor3):
        fused_tensor1 = self.linear1(tensor1)
        fused_tensor2 = self.linear2(tensor2)
        fused_tensor3 = self.linear3(tensor3)

        fused_output = fused_tensor1 + fused_tensor2 + fused_tensor3

        return fused_output
    
class ReshapeBlock(nn.Module):
    def __init__(self, channels = 32, length = 450 ):
        super(ReshapeBlock, self).__init__()        
        self.fc = nn.Linear(in_features=channels*length, out_features=64)  
        self.decode = nn.Linear(in_features=64, out_features=1800)  

    def forward(self, x):        
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = self.decode(x)        
        return x