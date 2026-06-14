"""Dual-head rollout loss, gradient, split, and checkpoint tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]
_NETWORK = _REPO / "python" / "network"
_NETWORK_TESTS = _NETWORK / "pytests"
for path in (_REPO / "python", _NETWORK, _NETWORK_TESTS, _REPO / "cpp" / "build"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import kumpel_network_json_fixtures as fixtures
from inference_models import create_self_play_models
from training.rollout_data import (
    ActionEstimate,
    OutcomeCounts,
    RolloutShard,
    RootState,
    TargetChoiceEstimate,
    TargetContextEstimate,
)
from training.trainer import (
    CheckpointMetadata,
    ModelStore,
    RolloutTrainer,
    split_roots,
)


def _root(root_id: str = "f" * 64) -> RootState:
    return RootState(
        root_id=root_id,
        state_bytes=fixtures.load_smoke_game_state_bytes(),
        deck_list1={},
        deck_list2={},
        player1_name="player1",
        player2_name="player2",
        player_name="player1",
        interaction_bytes=fixtures.load_smoke_game_interaction_bytes(),
        ply_from_end=1,
        phase="terminal",
        action_estimates=[
            ActionEstimate(0, OutcomeCounts(wins=7, losses=1)),
            ActionEstimate(1, OutcomeCounts(wins=1, losses=7)),
        ],
        target_contexts=[
            TargetContextEstimate(
                interaction_index=0,
                forced_prefix_deck_ids=[],
                candidate_deck_ids=[8, 10],
                include_stop_token=False,
                choices=[
                    TargetChoiceEstimate(0, OutcomeCounts(wins=6, losses=2)),
                    TargetChoiceEstimate(1, OutcomeCounts(wins=2, losses=6)),
                ],
            )
        ],
    )


def test_dual_head_training_updates_interaction_and_target_heads() -> None:
    network, selector, device = create_self_play_models(torch.device("cpu"))
    trainer = RolloutTrainer(network, selector, device, learning_rate=1e-3)

    result = trainer.train_step([_root()])

    assert result.target_value is not None
    assert result.target_policy is not None
    network_grads = {
        name: parameter.grad
        for name, parameter in network.named_parameters()
        if "value_head" in name or "policy_head" in name
    }
    selector_grads = {
        name: parameter.grad
        for name, parameter in selector.named_parameters()
        if "value_head" in name or "policy_head" in name
    }
    assert network_grads and all(grad is not None for grad in network_grads.values())
    assert selector_grads and all(grad is not None for grad in selector_grads.values())


def test_batch_without_target_contexts_omits_target_losses() -> None:
    network, selector, device = create_self_play_models(torch.device("cpu"))
    trainer = RolloutTrainer(network, selector, device)
    root = _root()
    root.target_contexts = []

    result = trainer.train_step([root])

    assert result.target_value is None
    assert result.target_policy is None


def test_calibration_metrics_include_interaction_and_target_brier() -> None:
    network, selector, device = create_self_play_models(torch.device("cpu"))
    trainer = RolloutTrainer(network, selector, device)

    metrics = trainer.calibration_metrics([_root()], batch_size=1)

    assert metrics["interaction_brier"] is not None
    assert metrics["target_brier"] is not None
    assert 0.0 <= metrics["interaction_brier"] <= 1.0
    assert 0.0 <= metrics["target_brier"] <= 1.0


def test_stable_split_never_leaks_roots() -> None:
    roots = [_root(f"{index:064x}") for index in range(100)]
    training, validation = split_roots([RolloutShard(1, "c", roots)])

    training_ids = {root.root_id for root in training}
    validation_ids = {root.root_id for root in validation}
    assert training_ids.isdisjoint(validation_ids)
    assert training_ids | validation_ids == {root.root_id for root in roots}


def test_model_store_reject_path_does_not_replace_champion(tmp_path: Path) -> None:
    store = ModelStore(tmp_path)
    network, selector, device = create_self_play_models(torch.device("cpu"))
    store.save_checkpoint(
        store.champion_path,
        network,
        selector,
        metadata=CheckpointMetadata(0, "champion", "champion"),
    )
    champion_bytes = store.champion_path.read_bytes()
    with torch.no_grad():
        next(network.parameters()).add_(1.0)
    store.save_checkpoint(
        store.candidate_path,
        network,
        selector,
        metadata=CheckpointMetadata(1, "candidate", "candidate"),
    )

    assert store.champion_path.read_bytes() == champion_bytes


def test_old_checkpoint_format_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "old.pt"
    torch.save({"network": {}}, path)
    network, selector, device = create_self_play_models(torch.device("cpu"))

    with pytest.raises(ValueError, match="Unsupported checkpoint format"):
        ModelStore.load_checkpoint(path, network, selector, device)


def test_promotion_archives_previous_champion(tmp_path: Path) -> None:
    store = ModelStore(tmp_path)
    network, selector, _ = create_self_play_models(torch.device("cpu"))
    store.save_checkpoint(
        store.champion_path,
        network,
        selector,
        metadata=CheckpointMetadata(0, "old", "champion"),
    )
    with torch.no_grad():
        next(network.parameters()).add_(1.0)
    store.save_checkpoint(
        store.candidate_path,
        network,
        selector,
        metadata=CheckpointMetadata(1, "new", "candidate"),
    )

    store.promote(
        store.candidate_path,
        previous_champion_id="old",
        iteration=1,
        history_limit=4,
        manifest={"iteration": 1},
    )

    assert list(store.history_dir.glob("champion_*.pt"))
    loaded_network, loaded_selector, device = create_self_play_models(
        torch.device("cpu")
    )
    metadata = store.load_checkpoint(
        store.champion_path,
        loaded_network,
        loaded_selector,
        device,
    )
    assert metadata.champion_id == "new"
    assert metadata.role == "champion"


def test_soft_rollout_fixture_learns_action_ordering() -> None:
    torch.manual_seed(4)
    network, selector, device = create_self_play_models(torch.device("cpu"))
    trainer = RolloutTrainer(network, selector, device, learning_rate=1e-3)
    root = _root()
    root.target_contexts = []
    root.action_estimates = [
        ActionEstimate(0, OutcomeCounts(wins=15, losses=1)),
        ActionEstimate(1, OutcomeCounts(wins=1, losses=15)),
    ]

    for _ in range(10):
        trainer.train_step([root])

    network.eval()
    with torch.inference_mode():
        scores, _, _, _ = network.forward(
            root.state_bytes,
            root.interaction_bytes,
        )

    assert scores.value_logits[0] > scores.value_logits[1]
    assert scores.policy_logits[0] > scores.policy_logits[1]
