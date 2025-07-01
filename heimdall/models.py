"""
This module defines the HEIMDALL seismic signal processing framework,
which includes components for feature extraction, graph-based encoding,
detection, picking, and 3D localization.

Modules:
    - FrontEndCNN: Compact 1D CNN block with LayerNorm to process time series.
    - TemporalTransformer: Transformer-based encoder attending across time.
    - HeimdallEncoder: Full encoder stack combining CNN, Transformer,
      Graph Neural Network, and feature fusion.
    - HeimdallDetectorPicker: Detection and picking head predicting event
      probability and phase picks (P, S) from encoded features.
    - HeimdallLocator: Location head producing 2D PDF-like location images
      and direct coordinate regression using attention-pooling of nodes.
    - HEIMDALL: Full model combining encoder, detector, and locator.

Classes:
    FrontEndCNN(nn.Module)
        1D CNN frontend with optional downsampling and LayerNorm normalization.

    TemporalTransformer(nn.Module)
        TransformerEncoder applied along the temporal dimension of features.

    HeimdallEncoder(nn.Module)
        Combines CNN, Transformer, GCN, GraphTransformer, and pooling
        mechanisms to produce enriched station-level features.

    HeimdallDetectorPicker(nn.Module)
        Decodes encoded features to detect event presence and P/S picks,
        using LSTM and transposed convolutions to produce final classification maps.

    HeimdallLocator(nn.Module)
        Computes 3 location probability maps and regresses (x, y, z)
        coordinates from pooled graph-level embeddings.

    HEIMDALL(nn.Module)
        Full framework combining encoder, detector, and locator components.

Dependencies:
    torch, torch.nn, torch.nn.functional, torch_geometric.nn,
    torch_scatter, torchvision, math, heimdall

Example:
    >>> model = HEIMDALL(input_num_channels=3, input_features=501, stations_coords=coords_tensor)
    >>> det_out, loc_out, coords = model(x, edge_index, edge_attr, batch)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphNorm, TransformerConv
import math
from torch_scatter import scatter_softmax, scatter_sum, scatter_max, scatter_min  # scatter_mean
from heimdall import custom_logger as CL

logger = CL.init_logger("MODELS", lvl="INFO")


#######################################################################
# FRONT-END CNN
#######################################################################

class FrontEndCNN(nn.Module):
    """
    FrontEndCNN

    A compact 1D CNN block with LayerNorm normalization, which processes
    multi-channel 1D inputs (waveform components) using stacked
    convolutional layers. Optionally downsamples the temporal dimension
    through stride, followed by LayerNorm and ReLU activations.
    Finally, it applies an adaptive average pooling to produce a
    fixed temporal dimension.

    Args:
        in_channels (int): Number of input channels (e.g., 3-component waveform).
        base_filters (int): Number of filters in the first convolutional layer.
        depth (int): Number of stacked convolutional layers.
        kernel_size (int): Kernel size for each convolution.
        temporal_dim (int): Output temporal dimension after adaptive pooling.

    Shape:
        - Input: [batch_size, in_channels, time_length]
        - Output: [batch_size, out_channels, temporal_dim]
    """
    def __init__(self,
                 in_channels:   int = 3,     # e.g. 3‑component waveform
                 base_filters:  int = 16,    # filters in the first conv
                 depth:         int = 2,     # how many conv layers
                 kernel_size:   int = 7,
                 temporal_dim:  int = 64):   # length after pooling
        super().__init__()

        self.convs = nn.ModuleList()
        self.lns   = nn.ModuleList()

        curr_in = in_channels
        for i in range(depth):
            curr_out = base_filters * (2 ** i)
            stride   = 1 if i == 0 else 2      # optional down‑sampling

            self.convs.append(
                nn.Conv1d(
                    in_channels=curr_in,
                    out_channels=curr_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2   # “same” padding
                )
            )
            # LayerNorm will see tensor shape [B, T, C] → last dim = C
            self.lns.append(nn.LayerNorm(curr_out))
            curr_in = curr_out

        # reduce time dimension to `temporal_dim` but keep it explicit
        self.global_pool = nn.AdaptiveAvgPool1d(temporal_dim)

    def forward(self, x):
        """
        x: [B, in_channels, time_len]
        """
        out = x
        for conv, ln in zip(self.convs, self.lns):
            out = conv(out)              # [B, C, T]
            out = out.permute(0, 2, 1)   # [B, T, C]  (channel last)
            out = ln(out)                # LayerNorm over C
            out = F.relu(out)
            out = out.permute(0, 2, 1)   # back to [B, C, T]

        out = self.global_pool(out)      # [B, C_final, temporal_dim]
        return out


#######################################################################
# TEMPORAL TRANSFORMER
#######################################################################

class TemporalTransformer(nn.Module):
    """
    TemporalTransformer

    Applies a standard TransformerEncoder over the temporal dimension of input features. Designed to process sequence data where the temporal axis is treated as the sequence dimension for attention. Adds learnable positional encodings and outputs feature maps with the same shape as input.

    Args:
        d_model (int): Dimensionality of input feature embedding.
        seq_len (int): Length of the temporal sequence.
        nhead (int): Number of attention heads.
        num_layers (int): Number of TransformerEncoder layers.
        dim_feedforward (int): Hidden dimension of feedforward sublayers.
        dropout (float): Dropout rate applied in Transformer layers.

    Shape:
        - Input: [batch_size, d_model, seq_len]
        - Output: [batch_size, d_model, seq_len]
    """

    def __init__(self, d_model, seq_len, nhead=4, num_layers=2,
                 dim_feedforward=256, dropout=0.2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # so shape is [batch, seq, embed]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Create & store the sinusoidal positional embeddings as a buffer
        pe = self.sinusoidal_position_encoding(seq_len, d_model)  # [seq_len, d_model]
        pe = pe.unsqueeze(0)                                      # [1, seq_len, d_model]
        self.register_buffer("positional_encoding", pe)

    def sinusoidal_position_encoding(self, seq_len, d_model):
        position = torch.arange(seq_len).unsqueeze(1)  # [seq_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # [seq_len, d_model]

    def forward(self, x):
        """
        x shape: [B, d_model, T']
        returns: [B, d_model, T']

        We take care here of handling the time-vector correctly
        """

        # We swhitch position to treat the TimeDimension (T') as the sequence
        x = x.transpose(1, 2)  # => [B, T', d_model]

        # # Add positional encoding
        # T = x.size(1)
        # x = x + self.positional_encoding[:, :T, :]

        x = x + self.positional_encoding

        # Forward through the TransformerEncoder
        x = self.transformer(x)  # => [B, T', d_model]

        # Transpose back to [B, d_model, T]
        x = x.transpose(1, 2)
        return x


################################################################################
# HEIMDALL ENCODER
################################################################################

class HeimdallEncoder(nn.Module):
    """
    HeimdallEncoder

    Full encoder module combining CNN frontend, Temporal Transformer,
    Graph Neural Network layers, and feature fusion mechanisms.
    Processes multi-station time series data, projects them to graph
    node features, applies graph message passing, and outputs enhanced
    temporal features. Supports different pooling mechanisms over
    time ('attn', 'mean', 'none').

    Args:
        input_num_channels (int): Number of input channels per node (waveform components).
        input_features (int): Length of input time series.
        hidden_features (int): Number of hidden features in GCN and graph transformer layers.
        cnn_out_channels (int): Output channels from the CNN frontend.
        cnn_temporal_dim (int): Temporal dimension after CNN processing.
        stations_coords (torch.Tensor): Tensor containing station coordinates (x, y, z).
        pool (str): Pooling method across time ('attn', 'mean', 'none').

    Shape:
        - Input: [B*N, input_num_channels, input_features]
        - Output: [B*N, cnn_out_channels, cnn_temporal_dim]
    """
    def __init__(self,
                 input_num_channels: int,
                 input_features: int,
                 hidden_features: int,
                 cnn_out_channels: int,
                 cnn_temporal_dim: int,
                 stations_coords: torch.Tensor,
                 pool: str = "attn"):           # "mean" | "attn" | "none"

        super().__init__()

        self.register_buffer('station_coords', stations_coords)
        assert pool in {"mean", "attn", "none"}
        self.pool = pool
        logger.warning("Using method: %s for graph-message-passing" % self.pool)

        self.hidden_features = hidden_features
        self.cnn_out_channels = cnn_out_channels
        self.drop = nn.Dropout(0.2)

        # 1) CNN front‑end
        self.frontend = FrontEndCNN(
            in_channels=input_num_channels,
            base_filters=cnn_out_channels // 2,
            depth=2,
            kernel_size=7,
            temporal_dim=cnn_temporal_dim
        )

        # 2) Temporal Transformer
        self.temporal_transformer = TemporalTransformer(
            d_model=cnn_out_channels,
            seq_len=cnn_temporal_dim,
            nhead=8,
            num_layers=4,
            dim_feedforward=256,
            dropout=0.2
        )

        # (optional) attention pooling weights over time
        if pool == "attn":
            self.time_attn = nn.Sequential(
                nn.Conv1d(cnn_out_channels, cnn_out_channels // 2, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(cnn_out_channels // 2, 1, kernel_size=1)         # logits over T′
            )

        # 3) Graph stack
        self.gcn1 = GCNConv(cnn_out_channels + 3, hidden_features)
        self.gn1 = GraphNorm(hidden_features)
        self.graph_trans = TransformerConv(
            in_channels=hidden_features,
            out_channels=hidden_features,
            heads=8,
            concat=False,
            dropout=0.2,
            edge_dim=1
        )

        # 4) post‑graph feed‑forward (adds non‑linearity) + residual
        self.post_graph_ffn = nn.Sequential(
            nn.Linear(hidden_features, hidden_features * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_features * 4, hidden_features)
        )

        # 5) projection back to temporal channel dimension
        self.graph_proj = nn.Linear(hidden_features, cnn_out_channels)

    # ----------------------------------------------------------------------
    # helper: repeat edge_index T times (for “pool = 'none'”)
    # ----------------------------------------------------------------------
    def __repeat_edge_index(self, edge_index, repeats: int, nodes_per_slice: int):
        """
        edge_index: Tensor [2, E]        – edges for ONE time slice
        repeats:     int                 – number of time steps (T′)
        nodes_per_slice: int             – #nodes in one slice (= B * N)
        returns: new_edge_index [2, E * repeats]
        """
        ei_list = []
        for t in range(repeats):
            offset = t * nodes_per_slice
            ei_list.append(edge_index + offset)
        return torch.cat(ei_list, dim=1)

    def forward(self, x, edge_index, edge_weights):
        """
        x           : [B*N, C, F]
        edge_index  : [2, E]
        edge_weights: [E] or [E,1]
        returns     : [B*N, cnn_out_channels, T′]
        """
        # ---------- 1 · temporal path ---------------------------------
        x = self.frontend(x)                     # [B*N, C_out, T′]
        x_t = self.temporal_transformer(x)       # [B*N, C_out, T′]

        # ---------- 2 · prepare graph input ---------------------------
        if self.pool == "none":
            Bn, C, T = x_t.shape
            x_graph = x_t.permute(0, 2, 1).reshape(Bn * T, C)   # [B*N*T, C_out]
            edge_index_rep = self.__repeat_edge_index(edge_index, T, Bn)
            edge_w_rep = edge_weights.repeat(T)

            # ---- NEW: repeat coords for every time slice -------------
            coords_rep = self.station_coords.repeat(Bn // self.station_coords.size(0), 1)
            coords_rep = coords_rep.repeat_interleave(T, dim=0)        # [B*N*T, 3]

        elif self.pool == "attn":
            Bn, C, T = x_t.shape
            logits = self.time_attn(x_t)                           # [B*N,1,T′]
            alpha = torch.softmax(logits, dim=-1)                 # [B*N,1,T′]
            x_graph = (x_t * alpha).sum(dim=2)                     # [B*N, C_out]
            edge_index_rep, edge_w_rep = edge_index, edge_weights

            # ---- NEW: one coord row per station -----------------------
            coords_rep = self.station_coords.repeat(Bn // self.station_coords.size(0), 1)  # [B*N,3]
            # print("coords_rep shape: ", coords_rep.shape)

        else:  # "mean"
            Bn, C, T = x_t.shape
            x_graph = x_t.mean(dim=2)                              # [B*N, C_out]
            edge_index_rep, edge_w_rep = edge_index, edge_weights

            # ---- NEW: one coord row
            coords_rep = self.station_coords.repeat(Bn // self.station_coords.size(0), 1)

        # ---------- concatenate coordinates ---------------------------
        x_graph = torch.cat([x_graph, coords_rep], dim=1)               # +3 features

        # ---------- 3 · graph message passing ------------------------
        x_gcn = self.gcn1(x_graph, edge_index_rep, edge_w_rep)
        x_gcn = self.gn1(x_gcn).relu_()
        x_gcn = self.drop(x_gcn)

        # residual = x_gcn
        edge_attr = edge_w_rep.unsqueeze(-1) if edge_w_rep is not None else None
        x_gcn = self.graph_trans(x_gcn, edge_index_rep, edge_attr=edge_attr)
        # x_gcn = x_gcn + residual                          # first residual
        x_gcn = x_gcn + self.post_graph_ffn(x_gcn)        # FFN + residual

        # ---------- 4 · reshape back if we flattened time ------------
        if self.pool == "none":
            x_gcn = x_gcn.view(Bn, T, -1).permute(0, 2, 1)   # [B*N, hidden, T′]
        else:
            x_gcn = x_gcn.unsqueeze(-1)                      # [B*N, hidden, 1]

        # ---------- 5 · project & fuse with temporal stream ----------
        #   project last dim (hidden) → cnn_out_channels
        g_proj = self.graph_proj(x_gcn.permute(0, 2, 1))     # [B*N, T′/1, C_out]
        g_proj = g_proj.permute(0, 2, 1)                     # [B*N, C_out, T′]
        x_out = x_t + g_proj                                 # simple residual sum
        return x_out


#######################################################################
# DETECTION head
#######################################################################

class HeimdallDetectorPicker(nn.Module):
    """
    Detection and picking head for Heimdall model.

    This module processes the hidden representation from the encoder and
    produces three outputs over the time axis: overall detection score,
    and refined P and S phase pick probabilities. Internally, it applies
    a BiLSTM for temporal smoothing and transposed convolution layers to
    upsample the sequence length to the desired output resolution.

    Args:
        output_channels (int): Number of output channels (expected to be 3 for [event, P, S]).
        output_features (int): Target temporal length of the output sequence (e.g., 501).
        hidden_features (int): Number of input channels from the encoder and LSTM hidden size.
        cnn_temporal_dim (int): Temporal dimension of CNN output from the encoder.

    Returns:
        torch.Tensor: Output tensor of shape [B*N, 3, output_features], where channels correspond
        to detection, P-phase, and S-phase probabilities respectively.
    """
    def __init__(self,
                 output_channels,
                 output_features,
                 hidden_features,
                 cnn_temporal_dim):  # T' from encoder
        super().__init__()

        self.output_channels = output_channels
        self.output_features = output_features
        self.hidden_features = hidden_features
        self.cnn_temporal_dim = cnn_temporal_dim

        # BiLSTM across time
        self.lstm = nn.LSTM(
            input_size=hidden_features,
            hidden_size=hidden_features // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Upsampling stack
        self.transconv1 = nn.ConvTranspose1d(hidden_features, hidden_features // 2, 4, 2, 1)
        # self.bn1 = nn.BatchNorm1d(hidden_features // 2)
        self.bn1 = nn.GroupNorm(4, hidden_features // 2)

        self.transconv2 = nn.ConvTranspose1d(hidden_features // 2, cnn_temporal_dim, 4, 2, 1)
        # self.bn2 = nn.BatchNorm1d(cnn_temporal_dim)
        self.bn2 = nn.GroupNorm(4, cnn_temporal_dim)

        self.transconv3 = nn.ConvTranspose1d(cnn_temporal_dim, cnn_temporal_dim // 2, 4, 2, 1)
        # self.bn3 = nn.BatchNorm1d(cnn_temporal_dim // 2)
        self.bn3 = nn.GroupNorm(4, cnn_temporal_dim // 2)

        self.conv_final = nn.Conv1d(cnn_temporal_dim // 2, 3, kernel_size=12, stride=1, padding=0)

    def forward(self, x):
        """
        x: [B*N, hidden_features, T']
        returns: [B*N, 3, 501]
        """
        # LSTM expects [B*N, T', D]
        x = x.transpose(1, 2)         # → [B*N, T', D]
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)         # → [B*N, D, T']

        # Transposed convs to upsample time
        x = F.relu(self.bn1(self.transconv1(x)))     # upsample step 1
        x = F.relu(self.bn2(self.transconv2(x)))     # upsample step 2
        x = F.relu(self.bn3(self.transconv3(x)))     # upsample step 3
        x = self.conv_final(x)                       # final conv (compress to 3 channels)

        # Ensure output length = 501
        if x.shape[-1] > self.output_features:
            x = x[..., :self.output_features]  # crop extra time steps
        # elif x.shape[-1] < self.output_features:
        #     x = F.interpolate(x, size=self.output_features, mode='linear', align_corners=True)
        else:
            raise ValueError("Final shape is shorter than output ... check model!")

        # Gating mechanism
        det = torch.sigmoid(x[:, 0, :])              # [B*N, 501]
        p = torch.sigmoid(x[:, 1, :]) * det
        s = torch.sigmoid(x[:, 2, :]) * det
        return torch.stack([det, p, s], dim=1)       # [B*N, 3, 501]


#######################################################################
# LOCATOR head
#######################################################################

class HeimdallLocator(nn.Module):
    """
    Location head for Heimdall model.

    This module predicts three location probability maps (representing
    2D projections in different planes) and regresses continuous (x, y, z)
    coordinates of the source. It supports hybrid attention mechanisms
    using learned attention or externally provided attention weights
    (e.g., derived from detector pick times).

    Args:
        hidden_features (int): Size of the hidden feature dimension from the encoder.
        location_output_sizes (List[Tuple[int, int]]): List of (height, width) tuples
            specifying the shape of each output location probability map.
        rank (int, optional): Intermediate dimension in low-rank location head projection.
        hybrid_attention (bool, optional): If True, combines learned attention with
            external attention weights.

    Returns:
        Tuple[List[torch.Tensor], torch.Tensor]:
            - List of 3 tensors representing 2D location probability maps of shape [B, H, W].
            - Tensor of shape [B, 3] with continuous (x, y, z) coordinate predictions.
    """
    def __init__(self,
                 hidden_features: int,
                 location_output_sizes,
                 rank: int = 128,
                 hybrid_attention: bool = False):  # Optional: combine learned + pick attention
        super().__init__()

        self.hybrid_attention = hybrid_attention

        # Ensure all output sizes are torch.Size objects
        self.location_output_sizes = [
            torch.Size(s) if not isinstance(s, torch.Size) else s
            for s in location_output_sizes
        ]
        self.total_pixels = sum(s.numel() for s in self.location_output_sizes)

        # 1-query learned attention weight (optional)
        self.score = nn.Linear(hidden_features, 1)

        # Low-rank location head
        self.loc_head = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_features, rank, bias=False),   # [hidden → rank]
            nn.ReLU(inplace=True),
            nn.Linear(rank, self.total_pixels)               # [rank → flattened image space]
        )

        # Side regression head for (x, y, z)
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, 3),
            nn.Sigmoid()  # output in [0, 1] space
        )

    def forward(self, x, batch, attention_weights=None):
        """
        Args:
            x: [B*N, hidden, T′]       – time-collapsed station embeddings
            batch: [B*N]               – graph ID per node
            attention_weights: [B*N]  – optional external attention (e.g., from pick times)

        Returns:
            imgs: 3 projected 2D PDFs
            coords: [B, 3] direct (x, y, z) coordinates
        """
        if batch.device != x.device:
            batch = batch.to(x.device)

        # Collapse time → station embedding: [B*N, hidden]
        h = x.mean(dim=2)

        # ---- Attention weighting over stations ----
        if attention_weights is not None:
            # Optionally combine with learned attention
            if self.hybrid_attention:
                learned = self.score(h).squeeze(-1)  # [B*N]
                alpha = learned + attention_weights
            else:
                alpha = attention_weights  # external weights only
        else:
            alpha = self.score(h).squeeze(-1)  # fallback to learned attention

        # Normalize per graph
        alpha = scatter_softmax(alpha, batch, dim=0)

        # Pool node features into graph summary using attention
        pooled = scatter_sum(h * alpha.unsqueeze(-1), batch, dim=0)  # [B, hidden]

        # ---- Predict location maps ----
        flat = torch.sigmoid(self.loc_head(pooled))  # [B, total_pixels]

        imgs, idx = [], 0
        for size in self.location_output_sizes:
            n = size.numel()
            imgs.append(flat[:, idx:idx + n].view(-1, *size))
            idx += n

        # ---- Predict direct (x, y, z) coordinates ----
        coords = self.coord_head(pooled)  # [B, 3]

        return (imgs, coords)


#######################################################################
# COMPLETE MODEL DEFINITION
#######################################################################

class HEIMDALL(nn.Module):
    """
    The main HEIMDALL model integrating detection, picking, and location modules
    for seismic event analysis.

    This composite model includes:
    - A shared encoder (CNN → Temporal Transformer → GCN → Graph Transformer)
      that processes graph-structured multi-station time series data.
    - A detection head producing [event, P, S] probability sequences.
    - A locator head generating 2D location probability maps and
      direct (x, y, z) coordinate estimates.

    Args:
        input_num_channels (int): Number of input channels in time series
                                  data (e.g., 3 for 3-component signals).
        input_features (int): Length of the input time series sequence.
        hidden_features (int): Hidden dimension size for graph layers
                              and downstream modules.
        cnn_out_channels (int): Number of output channels from CNN frontend.
        cnn_temporal_dim (int): Temporal dimension of CNN/Transformer output.
        stations_coords (torch.Tensor): Tensor of shape [num_stations, 3]
                                        containing station coordinates.
        location_output_sizes (List[Tuple[int, int]]): List of (height, width)
            tuples specifying the shape of each output 2D location probability map.

    Returns:
        Tuple:
            - det_out (torch.Tensor): Tensor of shape [B*N, 3, 501]
                                      with detection, P, and S probabilities.
            - loc_out (List[torch.Tensor]): List of 3 tensors [B, H, W]
                                            representing location probability maps.
            - coords (torch.Tensor): Tensor of shape [B, 3] with direct
                                     (x, y, z) coordinates.
    """
    def __init__(self,
                 input_num_channels=3,     # e.g. 3C signal
                 input_features=501,       # length of time series
                 hidden_features=256,      # GCN output / LSTM input
                 cnn_out_channels=128,     # CNN output channels
                 cnn_temporal_dim=88,      # CNN/Transformer time output
                 #
                 stations_coords=None,
                 location_output_sizes=[
                     (304, 330), (304, 151), (330, 151)]):

        super().__init__()

        logger.info(f"input_num_channels: {input_num_channels!r}")
        logger.info(f"input_features: {input_features!r}")
        logger.info(f"hidden_features: {hidden_features!r}")
        logger.info(f"cnn_out_channels: {cnn_out_channels!r}")
        logger.info(f"cnn_temporal_dim: {cnn_temporal_dim!r}")
        logger.info(f"location_output_sizes: {location_output_sizes!r}")

        # Shared encoder: CNN → Transformer → GCN → GraphTransformer
        self.encoder = HeimdallEncoder(
            input_num_channels=input_num_channels,
            input_features=input_features,
            hidden_features=hidden_features,
            cnn_out_channels=cnn_out_channels,
            cnn_temporal_dim=cnn_temporal_dim,
            stations_coords=stations_coords)

        # Detector: upsampling head for [event, P, S] classification
        self.detector = HeimdallDetectorPicker(
            output_channels=input_num_channels,
            output_features=input_features,
            hidden_features=cnn_out_channels,
            cnn_temporal_dim=cnn_temporal_dim)

        # Locator: predicts 3 location images per event
        self.locator = HeimdallLocator(
            location_output_sizes=location_output_sizes,
            hidden_features=cnn_out_channels)

    def forward(self, x, edge_index, edge_attr, batch):
        """
        x: [B*N, C, F]               ← time series input
        edge_index: [2, E]           ← graph connectivity
        edge_attr: [E] or [E, 1]     ← edge weights
        batch: [B*N]                 ← graph ID for each node

        Returns:
            det_out: [B*N, 3, 501]   ← detection (event, P, S)
            loc_out: 3 × [B, H, W]   ← location maps per graph
        """
        # === SHARED ENCODER ===
        enc_out = self.encoder(x, edge_index, edge_attr)  # [B*N, hidden, T']

        # === DETECTOR HEAD ===
        det_out = self.detector(enc_out)  # [B*N, 3, 501]

        # === PICK-BASED ATTENTION GUIDANCE ===
        p_scores = det_out[:, 1, :]  # [B*N, 501] → P-channel scores
        s_scores = det_out[:, 2, :]  # [B*N, 501] → S-channel scores

        # Soft argmin to infer pick time (weighted average)
        time_range = torch.arange(p_scores.size(1), device=p_scores.device).float()
        soft_picks_p = (p_scores * time_range).sum(dim=1) / (p_scores.sum(dim=1) + 1e-6)
        soft_picks_s = (s_scores * time_range).sum(dim=1) / (s_scores.sum(dim=1) + 1e-6)

        # Normalize per graph: early pick ⇒ higher weight
        min_p = scatter_min(soft_picks_p, batch, dim=0)[0][batch]
        max_p = scatter_max(soft_picks_p, batch, dim=0)[0][batch]
        weights_p = 1.0 - (soft_picks_p - min_p) / (max_p - min_p + 1e-6)

        min_s = scatter_min(soft_picks_s, batch, dim=0)[0][batch]
        max_s = scatter_max(soft_picks_s, batch, dim=0)[0][batch]
        weights_s = 1.0 - (soft_picks_s - min_s) / (max_s - min_s + 1e-6)

        # Combine pick weights (you can tune this weighting)
        weights = 1.0 * weights_p + 1.0 * weights_s  # or try different weigths
        weights = torch.clamp(weights, 0, 1)

        # === LOCATOR HEAD ===
        loc_out, coords = self.locator(enc_out, batch,
                                       attention_weights=weights)

        return (det_out, loc_out, coords)


# ==============================================================
# ==============================================================  REFERENCE
# ==============================================================

# -------- !!! NORMALIZATION / ACTIVATION / DROPOUT !!!
# The stride parameter controls the step size of the convolution kernel,
# effectively downsampling the output. This gradual reduction through
# learned convolutions retains far more information and spatial acuity
# than max pooling. Max Pooling only propagates the most prominent input.
# In contrast, convolutions can learn to combine features in complex, nonlinear ways.
# (W-F+2P)/S + 1  --> ie:  W=12 feature  F=4 filters P=2 padding S=3 stride --> outmap length:  5
