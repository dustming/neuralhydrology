from typing import Dict, List, Optional
import torch
import torch.nn as nn
import numpy as np

from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.routing import RoutingModule
from neuralhydrology.utils.config import Config


class RoutingLSTM(BaseModel):
    """Routing LSTM model for multi-subbasin hydrological modeling with routing.

    This model runs calibrated LSTMs on multiple subbasins and includes a routing component
    to route the outflow from upstream to downstream subbasins. Each subbasin has its own
    LSTM with potentially different parameters, and the routing is handled by a connectivity
    matrix that defines the flow relationships between subbasins.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    # specify submodules of the model that can later be used for finetuning
    module_parts = ["embedding_nets", "subbasin_lstms", "heads", "routing_module"]

    def __init__(self, cfg: Config):
        super(RoutingLSTM, self).__init__(cfg=cfg)

        # Get distributed model specific configurations
        self.num_subbasins = cfg.num_subbasins
        self.use_shared_lstm = getattr(cfg, "shared_lstm", False)

        # Initialize routing module
        self.routing_module = RoutingModule.from_config(cfg)

        # Initialize embedding networks for each subbasin (or shared)
        if getattr(cfg, "subbasin_specific_embedding", False):
            self.embedding_nets = nn.ModuleList(
                [InputLayer(cfg) for _ in range(self.num_subbasins)]
            )
        else:
            self.embedding_net = InputLayer(cfg)
            self.embedding_nets = None

        # Initialize LSTM networks for each subbasin
        if self.use_shared_lstm:
            # Single shared LSTM for all subbasins
            input_size = (
                self.embedding_net.output_size
                if self.embedding_nets is None
                else self.embedding_nets[0].output_size
            )
            self.shared_lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=cfg.hidden_size,
                num_layers=getattr(cfg, "num_layers", 1),
                dropout=(
                    getattr(cfg, "lstm_dropout", 0.0)
                    if getattr(cfg, "num_layers", 1) > 1
                    else 0.0
                ),
                batch_first=True,
            )
            self.subbasin_lstms = None
        else:
            # Individual LSTM for each subbasin
            input_size = (
                self.embedding_net.output_size
                if self.embedding_nets is None
                else self.embedding_nets[0].output_size
            )
            self.subbasin_lstms = nn.ModuleList(
                [
                    nn.LSTM(
                        input_size=input_size,
                        hidden_size=cfg.hidden_size,
                        num_layers=getattr(cfg, "num_layers", 1),
                        dropout=(
                            getattr(cfg, "lstm_dropout", 0.0)
                            if getattr(cfg, "num_layers", 1) > 1
                            else 0.0
                        ),
                        batch_first=True,
                    )
                    for _ in range(self.num_subbasins)
                ]
            )

        # Output dropout
        self.dropout = nn.Dropout(p=cfg.output_dropout)

        # Initialize prediction heads for each subbasin
        if getattr(cfg, "subbasin_specific_heads", False):
            self.heads = nn.ModuleList(
                [
                    get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)
                    for _ in range(self.num_subbasins)
                ]
            )
        else:
            self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)
            self.heads = None

        # Initialize parameters
        self._reset_parameters()

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass of the routing LSTM model.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary containing input data. Expected keys depend on the input layer configuration.
            For routing modeling, data should include subbasin-specific features.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing predictions and intermediate outputs.
        """

        # Handle input data - assume data contains subbasin-specific inputs
        if "x_d_subbasins" in data:
            # Subbasin-specific dynamic inputs: [batch_size, seq_len, num_subbasins, n_features]
            x_subbasins = data["x_d_subbasins"]
            batch_size, seq_len, num_subbasins, n_features = x_subbasins.shape
        else:
            # Standard input - replicate for each subbasin
            if self.embedding_nets is None:
                x_embedded = self.embedding_net(
                    data
                )  # [seq_len, batch_size, n_features]
            else:
                x_embedded = self.embedding_nets[0](
                    data
                )  # Use first embedding net as template

            seq_len, batch_size, n_features = x_embedded.shape
            # Replicate for all subbasins
            x_subbasins = x_embedded.unsqueeze(2).repeat(1, 1, self.num_subbasins, 1)
            x_subbasins = x_subbasins.permute(
                1, 0, 2, 3
            )  # [batch_size, seq_len, num_subbasins, n_features]
            num_subbasins = self.num_subbasins

        # Process each subbasin through embedding networks
        if self.embedding_nets is not None:
            embedded_outputs = []
            for i in range(num_subbasins):
                subbasin_data = {k: v for k, v in data.items()}
                if "x_d_subbasins" in data:
                    # Extract subbasin-specific data
                    subbasin_data["x_d"] = x_subbasins[:, :, i, :].permute(
                        1, 0, 2
                    )  # [seq_len, batch_size, n_features]

                embedded = self.embedding_nets[i](
                    subbasin_data
                )  # [seq_len, batch_size, embedded_size]
                embedded_outputs.append(embedded)

            # Stack embedded outputs: [seq_len, batch_size, num_subbasins, embedded_size]
            x_embedded_all = torch.stack(embedded_outputs, dim=2)
        else:
            # Use shared embedding
            if "x_d_subbasins" not in data:
                x_embedded_single = self.embedding_net(
                    data
                )  # [seq_len, batch_size, embedded_size]
                x_embedded_all = x_embedded_single.unsqueeze(2).repeat(
                    1, 1, num_subbasins, 1
                )
            else:
                # Process subbasin-specific data through shared embedding
                embedded_outputs = []
                for i in range(num_subbasins):
                    subbasin_data = {k: v for k, v in data.items()}
                    subbasin_data["x_d"] = x_subbasins[:, :, i, :].permute(1, 0, 2)
                    embedded = self.embedding_net(subbasin_data)
                    embedded_outputs.append(embedded)
                x_embedded_all = torch.stack(embedded_outputs, dim=2)

        # Process through LSTM networks
        seq_len, batch_size, num_subbasins, embedded_size = x_embedded_all.shape

        if self.use_shared_lstm:
            # Reshape for shared LSTM processing
            x_reshaped = x_embedded_all.permute(
                0, 2, 1, 3
            ).contiguous()  # [seq_len, num_subbasins, batch_size, embedded_size]
            x_reshaped = x_reshaped.view(
                seq_len, num_subbasins * batch_size, embedded_size
            )
            x_reshaped = x_reshaped.permute(
                1, 0, 2
            )  # [num_subbasins * batch_size, seq_len, embedded_size]

            lstm_output, (h_n, c_n) = self.shared_lstm(x_reshaped)

            # Reshape back: [batch_size, seq_len, num_subbasins, hidden_size]
            lstm_output = lstm_output.view(num_subbasins, batch_size, seq_len, -1)
            lstm_output = lstm_output.permute(1, 2, 0, 3)

            # Reshape hidden states
            h_n = h_n.view(
                -1, num_subbasins, batch_size, h_n.size(-1)
            )  # [num_layers, num_subbasins, batch_size, hidden_size]
            c_n = c_n.view(-1, num_subbasins, batch_size, c_n.size(-1))
        else:
            # Process each subbasin through its own LSTM
            lstm_outputs = []
            hidden_states = []
            cell_states = []

            for i in range(num_subbasins):
                subbasin_input = x_embedded_all[:, :, i, :].permute(
                    1, 0, 2
                )  # [batch_size, seq_len, embedded_size]
                lstm_out, (h_n, c_n) = self.subbasin_lstms[i](subbasin_input)
                lstm_outputs.append(lstm_out)
                hidden_states.append(h_n)
                cell_states.append(c_n)

            # Stack outputs: [batch_size, seq_len, num_subbasins, hidden_size]
            lstm_output = torch.stack(lstm_outputs, dim=2)
            h_n = torch.stack(
                hidden_states, dim=2
            )  # [num_layers, batch_size, num_subbasins, hidden_size]
            c_n = torch.stack(cell_states, dim=2)

        # Apply routing
        routing_result = self.routing_module(
            lstm_output.unsqueeze(-1)  # Add feature dimension for routing module
        )
        routed_output = routing_result["routed_flow"].squeeze(
            -1
        )  # Remove feature dimension

        # Apply dropout
        routed_output = self.dropout(routed_output)

        # Generate predictions through heads
        if self.heads is not None:
            # Subbasin-specific heads
            predictions = []
            for i in range(num_subbasins):
                subbasin_output = routed_output[
                    :, :, i, :
                ]  # [batch_size, seq_len, hidden_size]
                pred = self.heads[i](subbasin_output)
                predictions.append(pred)

            # Stack predictions and combine keys
            final_pred = {}
            for key in predictions[0].keys():
                final_pred[key] = torch.stack(
                    [pred[key] for pred in predictions], dim=2
                )  # [batch_size, seq_len, num_subbasins, n_outputs]
        else:
            # Shared head - process all subbasins
            batch_size, seq_len, num_subbasins, hidden_size = routed_output.shape
            routed_reshaped = routed_output.view(
                batch_size * seq_len * num_subbasins, hidden_size
            )
            pred = self.head(
                routed_reshaped.unsqueeze(1)
            )  # Add sequence dimension for head

            # Reshape predictions
            final_pred = {}
            for key, value in pred.items():
                # Reshape back to [batch_size, seq_len, num_subbasins, n_outputs]
                reshaped_value = value.squeeze(1).view(
                    batch_size, seq_len, num_subbasins, -1
                )
                final_pred[key] = reshaped_value

        # Add additional outputs
        routing_params = routing_result.get("routing_parameters", {})
        final_pred.update(
            {
                "lstm_output": lstm_output,
                "routed_output": routed_output,
                "routing_parameters": routing_params,
                "h_n": h_n.permute(
                    1, 0, 2, 3
                ),  # [batch_size, num_layers, num_subbasins, hidden_size]
                "c_n": c_n.permute(1, 0, 2, 3),
            }
        )

        return final_pred
