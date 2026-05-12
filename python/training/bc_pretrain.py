"""Behavioral cloning pre-trainer.

Loads demonstration data (.npz files from ``bc_collector``), trains an
SB3-compatible MlpPolicy via supervised learning, and saves in two formats:

1. **SB3-compatible .zip** -- ``PPO.load()`` can resume RL training from
   the BC-initialised weights.
2. **Standalone .pt** -- lightweight reference copy for the KL anchor
   callback to constrain policy drift during fine-tuning.

The network architecture mirrors SB3's default ``MlpPolicy``:
``Sequential(Linear, Tanh, Linear, Tanh)`` with a separate action head.
The default hidden sizes ``(64, 64)`` match SB3's default ``net_arch``.

Typical usage::

    from training.bc_pretrain import BCTrainer

    trainer = BCTrainer(obs_dim=116, act_dim=5)
    stats = trainer.train("data/demos.npz", epochs=50)

    # Save for PPO.load() warm-start
    trainer.save_as_sb3("runs/bc_model", env)

    # Save standalone reference for KL anchor
    trainer.save_reference("runs/bc_ref.pt")
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class BCTrainer:
    """Train a policy network via behavioral cloning on expert demonstrations.

    The feature extractor and action head are structured to match SB3's
    ``MlpPolicy`` so that trained weights can be copied directly into a
    PPO model via :meth:`save_as_sb3`.

    Parameters
    ----------
    obs_dim : int
        Observation vector dimensionality.
    act_dim : int
        Action vector dimensionality.
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
        act_dim: int,
        lr: float = 1e-3,
        hidden_sizes: tuple[int, ...] = (64, 64),
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self.lr = lr
        self.hidden_sizes = hidden_sizes

        # Build a feature extractor matching SB3's MlpPolicy architecture:
        #   Sequential(Linear(obs_dim, h0), Tanh, Linear(h0, h1), Tanh, ...)
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        self.feature_extractor = nn.Sequential(*layers).to(device)

        # Action head: Linear(last_hidden -> act_dim)
        # Matches SB3's `action_net` layer.
        self.action_head = nn.Linear(in_dim, act_dim).to(device)

        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters())
            + list(self.action_head.parameters()),
            lr=lr,
        )

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
        acts = torch.tensor(data["actions"], dtype=torch.float32)

        # Train/val split
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
            # --- Train ---
            self.feature_extractor.train()
            self.action_head.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch_obs, batch_acts in train_dl:
                batch_obs = batch_obs.to(self.device)
                batch_acts = batch_acts.to(self.device)

                features = self.feature_extractor(batch_obs)
                pred_acts = self.action_head(features)
                loss = nn.functional.mse_loss(pred_acts, batch_acts)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train = epoch_loss / max(n_batches, 1)
            stats["train_losses"].append(avg_train)

            # --- Validate ---
            self.feature_extractor.eval()
            self.action_head.eval()
            val_loss = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for batch_obs, batch_acts in val_dl:
                    batch_obs = batch_obs.to(self.device)
                    batch_acts = batch_acts.to(self.device)
                    features = self.feature_extractor(batch_obs)
                    pred_acts = self.action_head(features)
                    val_loss += nn.functional.mse_loss(pred_acts, batch_acts).item()
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
        """Predict action from observation.

        Parameters
        ----------
        obs : np.ndarray
            Observation array, either shape ``(obs_dim,)`` for a single
            sample or ``(batch, obs_dim)`` for a batch.

        Returns
        -------
        np.ndarray
            Predicted action(s). Shape ``(act_dim,)`` for single input,
            ``(batch, act_dim)`` for batch input.
        """
        self.feature_extractor.eval()
        self.action_head.eval()

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).to(self.device)
            single = obs_t.dim() == 1
            if single:
                obs_t = obs_t.unsqueeze(0)
            features = self.feature_extractor(obs_t)
            acts = self.action_head(features)

        result = acts.cpu().numpy()
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
                "action_head": self.action_head.state_dict(),
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
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
        trainer = cls(
            obs_dim=checkpoint["obs_dim"],
            act_dim=checkpoint["act_dim"],
            hidden_sizes=hidden_sizes,
            device=device,
        )
        trainer.feature_extractor.load_state_dict(checkpoint["feature_extractor"])
        trainer.action_head.load_state_dict(checkpoint["action_head"])
        return trainer

    def save_as_sb3(self, output_path: str, env) -> str:
        """Save as SB3-compatible PPO model.

        Creates a PPO instance with a matching ``net_arch``, copies the
        BC-trained weights into the policy and value networks, and saves
        via ``model.save()``. The resulting ``.zip`` can be loaded with
        ``PPO.load()`` and training can resume immediately.

        Parameters
        ----------
        output_path : str
            Path for the ``.zip`` file (without extension).
        env
            A Gymnasium environment instance (needed for PPO
            initialisation to infer observation/action spaces).

        Returns
        -------
        str
            The *output_path* argument (for chaining).
        """
        from stable_baselines3 import PPO

        # Create a fresh PPO with matching architecture.
        # net_arch must match hidden_sizes so the Sequential layers align.
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs={"net_arch": list(self.hidden_sizes)},
            verbose=0,
        )

        sb3_policy = model.policy

        # -----------------------------------------------------------------
        # SB3 MlpPolicy internal structure (with net_arch=[64, 64]):
        #
        #   mlp_extractor.policy_net = Sequential(
        #       (0): Linear(obs_dim, 64)
        #       (1): Tanh()
        #       (2): Linear(64, 64)
        #       (3): Tanh()
        #   )
        #   action_net = Linear(64, act_dim)
        #
        # Our BC model mirrors this exactly:
        #   feature_extractor = Sequential(
        #       (0): Linear(obs_dim, 64)
        #       (1): Tanh()
        #       (2): Linear(64, 64)
        #       (3): Tanh()
        #   )
        #   action_head = Linear(64, act_dim)
        #
        # State dict keys are identical (0.weight, 0.bias, 2.weight, 2.bias)
        # because Tanh layers have no parameters.
        # -----------------------------------------------------------------

        # Copy feature extractor weights -> SB3 policy_net
        sb3_policy.mlp_extractor.policy_net.load_state_dict(
            self.feature_extractor.state_dict()
        )

        # Copy action head weights -> SB3 action_net
        sb3_policy.action_net.load_state_dict(
            self.action_head.state_dict()
        )

        # Also copy to value network (warm-starts the value function)
        sb3_policy.mlp_extractor.value_net.load_state_dict(
            self.feature_extractor.state_dict()
        )

        model.save(output_path)
        print(f"Saved SB3-compatible model to {output_path}.zip")
        return output_path


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
        "--act-dim",
        type=int,
        required=True,
        help="Action vector dimensionality.",
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
        help="Path for the SB3-compatible .zip (without extension).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to env config (required for --output-sb3).",
    )
    parser.add_argument(
        "--scenario",
        default="cs_lite",
        help="Scenario name (required for --output-sb3).",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=None,
        help="Curriculum phase (required for --output-sb3).",
    )

    args = parser.parse_args()

    trainer = BCTrainer(
        obs_dim=args.obs_dim,
        act_dim=args.act_dim,
        lr=args.lr,
    )

    print(f"Training BC policy from {args.demos}")
    print(f"  obs_dim={args.obs_dim}, act_dim={args.act_dim}")
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
        if args.config is None:
            parser.error("--config is required when using --output-sb3")

        from training.ppo_trainer import _import_gym_class

        GymClass = _import_gym_class(args.scenario)
        env = GymClass(
            config_path=args.config,
            scenario=args.scenario,
            phase=args.phase,
        )
        os.makedirs(os.path.dirname(args.output_sb3) or ".", exist_ok=True)
        trainer.save_as_sb3(args.output_sb3, env)
        env.close()
