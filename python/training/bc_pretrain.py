"""Behavioral cloning pre-trainer (multi-head classification).

Loads demonstration data (.npz files from ``bc_collector``), trains a
multi-head classification network via supervised learning, and saves as a
standalone ``.pt`` checkpoint for KL anchor reference.

Each action head is a separate classification branch:
  ``Linear(hidden_dim, n_i)`` where ``n_i`` is the branch cardinality.
Loss is the sum of per-branch cross-entropy losses.

The feature extractor mirrors SB3's default ``MlpPolicy``:
``Sequential(Linear, Tanh, Linear, Tanh)``.
The default hidden sizes ``(64, 64)`` match SB3's default ``net_arch``.

Typical usage::

    from training.bc_pretrain import BCTrainer

    trainer = BCTrainer(obs_dim=116, branch_sizes=[12, 2, 2, 3])
    stats = trainer.train("data/demos.npz", epochs=50)

    # Save standalone reference for KL anchor
    trainer.save_reference("runs/bc_ref.pt")
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class BCTrainer:
    """Train a multi-head classification network via behavioral cloning.

    Each action head is a separate ``Linear(hidden_dim, n_i)`` branch.
    Training uses per-branch cross-entropy loss.

    Parameters
    ----------
    obs_dim : int
        Observation vector dimensionality.
    branch_sizes : list[int]
        Cardinality of each discrete action head (e.g. ``[12, 2, 2, 3]``).
    act_dim : int
        Deprecated, kept for backwards compat. Ignored when branch_sizes given.
    lr : float
        Adam learning rate (default 1e-3).
    hidden_sizes : tuple[int, ...]
        Hidden layer sizes. Each entry adds a ``Linear`` + ``Tanh`` block.
        The default ``(64, 64)`` matches SB3's default ``net_arch``.
    device : str
        Torch device (default ``"cpu"``).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int = 0,
        lr: float = 1e-3,
        hidden_sizes: tuple[int, ...] = (64, 64),
        device: str = "cpu",
        branch_sizes: list[int] | None = None,
    ):
        self.obs_dim = obs_dim
        self.device = device
        self.lr = lr
        self.hidden_sizes = hidden_sizes

        if branch_sizes is not None:
            self.branch_sizes = list(branch_sizes)
        elif act_dim > 0:
            self.branch_sizes = [act_dim]
        else:
            raise ValueError("Must provide branch_sizes or act_dim > 0")

        self.act_dim = len(self.branch_sizes)

        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        self.feature_extractor = nn.Sequential(*layers).to(device)

        self.action_heads = nn.ModuleList([
            nn.Linear(in_dim, n) for n in self.branch_sizes
        ]).to(device)

        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters())
            + list(self.action_heads.parameters()),
            lr=lr,
        )

    def _compute_loss(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Sum of per-branch cross-entropy losses."""
        total = torch.tensor(0.0, device=features.device)
        for i, head in enumerate(self.action_heads):
            logits = head(features)
            total = total + nn.functional.cross_entropy(logits, targets[:, i])
        return total

    def train(
        self,
        demos_path: str,
        epochs: int = 50,
        batch_size: int = 256,
        val_split: float = 0.1,
    ) -> dict:
        """Train on demonstrations. Returns training stats dict.

        Parameters
        ----------
        demos_path : str
            Path to a ``.npz`` file with keys ``observations`` and ``actions``.
        epochs : int
            Number of training epochs.
        batch_size : int
            Mini-batch size for SGD.
        val_split : float
            Fraction of data held out for validation (0.0 = no validation).

        Returns
        -------
        dict
            Keys: ``train_losses`` (list[float]), ``val_losses``
            (list[float]), ``best_val_loss`` (float).
        """
        data = np.load(demos_path)
        obs = torch.tensor(data["observations"], dtype=torch.float32)
        acts = torch.tensor(data["actions"], dtype=torch.long)

        n = len(obs)
        n_val = int(n * val_split)
        n_train = n - n_val
        perm = torch.randperm(n)
        train_idx, val_idx = perm[:n_train], perm[n_train:]

        train_ds = TensorDataset(obs[train_idx], acts[train_idx])
        val_ds = TensorDataset(obs[val_idx], acts[val_idx])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size)

        best_val_loss = float("inf")
        stats: dict = {"train_losses": [], "val_losses": []}

        for epoch in range(epochs):
            self.feature_extractor.train()
            self.action_heads.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch_obs, batch_acts in train_dl:
                batch_obs = batch_obs.to(self.device)
                batch_acts = batch_acts.to(self.device)

                features = self.feature_extractor(batch_obs)
                loss = self._compute_loss(features, batch_acts)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train = epoch_loss / max(n_batches, 1)
            stats["train_losses"].append(avg_train)

            self.feature_extractor.eval()
            self.action_heads.eval()
            val_loss = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for batch_obs, batch_acts in val_dl:
                    batch_obs = batch_obs.to(self.device)
                    batch_acts = batch_acts.to(self.device)
                    features = self.feature_extractor(batch_obs)
                    val_loss += self._compute_loss(features, batch_acts).item()
                    n_val_batches += 1

            avg_val = val_loss / max(n_val_batches, 1)
            stats["val_losses"].append(avg_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  Epoch {epoch + 1}/{epochs}: "
                    f"train_loss={avg_train:.6f}  val_loss={avg_val:.6f}"
                )

        stats["best_val_loss"] = best_val_loss
        return stats

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict action indices from observation (argmax per branch).

        Parameters
        ----------
        obs : np.ndarray
            Observation array, either shape ``(obs_dim,)`` for a single
            sample or ``(batch, obs_dim)`` for a batch.

        Returns
        -------
        np.ndarray
            Integer action indices. Shape ``(num_heads,)`` for single input,
            ``(batch, num_heads)`` for batch input.
        """
        self.feature_extractor.eval()
        self.action_heads.eval()

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device)
            single = obs_t.dim() == 1
            if single:
                obs_t = obs_t.unsqueeze(0)
            features = self.feature_extractor(obs_t)
            indices = [head(features).argmax(dim=-1) for head in self.action_heads]
            acts = torch.stack(indices, dim=-1)

        result = acts.cpu().numpy().astype(np.int64)
        if single:
            result = result.squeeze(0)
        return result

    def save_reference(self, path: str) -> None:
        """Save standalone PyTorch model for KL anchor reference.

        The checkpoint contains the feature extractor and action head
        state dicts along with dimension metadata, allowing reconstruction
        via :meth:`load_reference`.

        Parameters
        ----------
        path : str
            Output file path (typically ending in ``.pt``).
        """
        torch.save(
            {
                "feature_extractor": self.feature_extractor.state_dict(),
                "action_heads": self.action_heads.state_dict(),
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "branch_sizes": self.branch_sizes,
                "hidden_sizes": self.hidden_sizes,
            },
            path,
        )

    @classmethod
    def load_reference(cls, path: str, device: str = "cpu") -> BCTrainer:
        """Load a reference model saved with :meth:`save_reference`.

        Parameters
        ----------
        path : str
            Path to the ``.pt`` checkpoint.
        device : str
            Torch device to load onto.

        Returns
        -------
        BCTrainer
            A trainer instance with the loaded weights (ready for
            :meth:`predict`, not for further training).
        """
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        hidden_sizes = checkpoint.get("hidden_sizes", (64, 64))
        if isinstance(hidden_sizes, list):
            hidden_sizes = tuple(hidden_sizes)

        branch_sizes = checkpoint.get("branch_sizes")
        if branch_sizes is None:
            branch_sizes = [checkpoint["act_dim"]]

        trainer = cls(
            obs_dim=checkpoint["obs_dim"],
            hidden_sizes=hidden_sizes,
            device=device,
            branch_sizes=branch_sizes,
        )
        trainer.feature_extractor.load_state_dict(checkpoint["feature_extractor"])

        if "action_heads" in checkpoint:
            trainer.action_heads.load_state_dict(checkpoint["action_heads"])
        elif "action_head" in checkpoint:
            if len(trainer.action_heads) == 1:
                trainer.action_heads[0].load_state_dict(checkpoint["action_head"])
            else:
                print("WARNING: legacy single-head checkpoint, only loading feature extractor")

        return trainer

    def save_as_sb3(self, output_path: str, env) -> str:
        """Transfer BC weights into an SB3 PPO model and save as .zip."""
        from stable_baselines3 import PPO

        model = PPO("MlpPolicy", env, verbose=0)
        policy = model.policy

        with torch.no_grad():
            # Feature extractor → mlp_extractor.policy_net
            # BC: Sequential(Linear, Tanh, Linear, Tanh)
            # SB3: Sequential(Linear, Tanh, Linear, Tanh) at policy_net.0, policy_net.2
            bc_layers = [m for m in self.feature_extractor if isinstance(m, nn.Linear)]
            sb3_layers = [m for m in policy.mlp_extractor.policy_net if isinstance(m, nn.Linear)]
            for bc_l, sb3_l in zip(bc_layers, sb3_layers):
                sb3_l.weight.copy_(bc_l.weight)
                sb3_l.bias.copy_(bc_l.bias)

            # Action heads → action_net (concatenated)
            # SB3 action_net is single Linear(hidden, sum(branch_sizes))
            # Our action_heads are [Linear(hidden, n_i) for n_i in branch_sizes]
            weights = torch.cat([h.weight for h in self.action_heads], dim=0)
            biases = torch.cat([h.bias for h in self.action_heads], dim=0)
            policy.action_net.weight.copy_(weights)
            policy.action_net.bias.copy_(biases)

        model.save(output_path)
        print(f"  Saved SB3 model: {output_path}.zip")
        return f"{output_path}.zip"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Train a BC policy from demonstration data."
    )
    parser.add_argument(
        "--demos",
        required=True,
        help="Path to the .npz demo file (from bc_collector).",
    )
    parser.add_argument(
        "--obs-dim",
        type=int,
        required=True,
        help="Observation vector dimensionality.",
    )
    parser.add_argument(
        "--branch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Cardinality of each discrete action head (e.g. 12 2 2 3).",
    )
    parser.add_argument(
        "--act-dim",
        type=int,
        default=0,
        help="(Deprecated) Action vector dimensionality. Use --branch-sizes instead.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default 50).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Mini-batch size (default 256).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default 1e-3).",
    )
    parser.add_argument(
        "--output-ref",
        default=None,
        help="Path for the .pt reference model (for KL anchor).",
    )
    parser.add_argument(
        "--output-sb3",
        default=None,
        help="(Deprecated) Not supported for MultiDiscrete.",
    )

    args = parser.parse_args()

    if args.branch_sizes is None and args.act_dim <= 0:
        parser.error("Must provide --branch-sizes or --act-dim")

    trainer = BCTrainer(
        obs_dim=args.obs_dim,
        act_dim=args.act_dim,
        lr=args.lr,
        branch_sizes=args.branch_sizes,
    )

    print(f"Training BC policy from {args.demos}")
    print(f"  obs_dim={args.obs_dim}, branch_sizes={trainer.branch_sizes}")
    print(f"  epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print()

    stats = trainer.train(
        args.demos, epochs=args.epochs, batch_size=args.batch_size
    )

    print(f"\nFinal train_loss: {stats['train_losses'][-1]:.6f}")
    print(f"Best val_loss:    {stats['best_val_loss']:.6f}")

    if args.output_ref:
        os.makedirs(os.path.dirname(args.output_ref) or ".", exist_ok=True)
        trainer.save_reference(args.output_ref)
        print(f"Saved reference model to {args.output_ref}")

    if args.output_sb3:
        print(
            "WARNING: --output-sb3 is not supported for MultiDiscrete. "
            "Use PPO training from scratch."
        )
