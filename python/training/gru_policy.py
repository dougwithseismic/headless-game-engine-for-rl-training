"""GRU policy for sb3-contrib RecurrentPPO.

Wraps nn.GRU to match nn.LSTM's (h, c) state interface so it works
transparently with RecurrentPPO's sequence processing.
"""
import torch
import torch.nn as nn
from sb3_contrib.ppo_recurrent import MlpLstmPolicy


class _GruAsLstm(nn.Module):
    """nn.GRU wrapper that returns states in LSTM's (h, c) format."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x, hc=None):
        if hc is not None:
            h, _c = hc
            out, h_new = self.gru(x, h)
        else:
            out, h_new = self.gru(x)
        return out, (h_new, torch.zeros_like(h_new))


class MlpGruPolicy(MlpLstmPolicy):
    """Drop-in GRU replacement for MlpLstmPolicy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_size = self.lstm_actor.hidden_size
        num_layers = self.lstm_actor.num_layers
        actor_input = self.lstm_actor.input_size

        self.lstm_actor = _GruAsLstm(actor_input, hidden_size, num_layers)
        if self.lstm_critic is not None:
            critic_input = self.lstm_critic.input_size
            self.lstm_critic = _GruAsLstm(critic_input, hidden_size, num_layers)
