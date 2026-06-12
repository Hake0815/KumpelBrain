"""Dynamic-batching inference service for coordinator-mode self-play."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from network.kumpel_network import KumpelNetwork
from network.selector import Selector


def _to_cpu_if_needed(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.device.type == "cpu" else tensor.cpu()


@dataclass
class _PendingRequest:
    event: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None


@dataclass
class _MoveEvalRequest(_PendingRequest):
    state_bytes: bytes = b""
    interactions_bytes: list[bytes] = field(default_factory=list)


@dataclass
class _TargetScoreRequest(_PendingRequest):
    candidates: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    partial_selection: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    transformed_state: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    embedded_interaction: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    card_indices: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    include_stop_token: bool = False


class InferenceClient(ABC):
    @abstractmethod
    def evaluate_move(
        self,
        state_bytes: bytes,
        interactions_bytes: list[bytes],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ...

    @abstractmethod
    def score_targets(
        self,
        candidates: torch.Tensor,
        partial_selection: torch.Tensor,
        transformed_state: torch.Tensor,
        embedded_interaction: torch.Tensor,
        card_indices: torch.Tensor,
        include_stop_token: bool,
    ) -> torch.Tensor:
        ...

    def game_finished(self) -> None:
        pass


class DirectInferenceClient(InferenceClient):
    def __init__(
        self,
        network: KumpelNetwork,
        selector: Selector,
        compute_device: torch.device,
    ):
        self.network = network
        self.selector = selector
        self.compute_device = compute_device

    def evaluate_move(
        self,
        state_bytes: bytes,
        interactions_bytes: list[bytes],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.network.forward(state_bytes, interactions_bytes)

    def score_targets(
        self,
        candidates: torch.Tensor,
        partial_selection: torch.Tensor,
        transformed_state: torch.Tensor,
        embedded_interaction: torch.Tensor,
        card_indices: torch.Tensor,
        include_stop_token: bool,
    ) -> torch.Tensor:
        if candidates.device != self.compute_device:
            candidates = candidates.to(self.compute_device)
            partial_selection = partial_selection.to(self.compute_device)
            transformed_state = transformed_state.to(self.compute_device)
            embedded_interaction = embedded_interaction.to(self.compute_device)
            card_indices = card_indices.to(self.compute_device)
        return self.selector(
            candidates,
            partial_selection,
            transformed_state,
            embedded_interaction,
            card_indices,
            include_stop_token=include_stop_token,
        )


class BatchedInferenceClient(InferenceClient):
    def __init__(self, service: BatchedInferenceService):
        self._service = service

    def evaluate_move(
        self,
        state_bytes: bytes,
        interactions_bytes: list[bytes],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._service.submit_move_eval(state_bytes, interactions_bytes)

    def score_targets(
        self,
        candidates: torch.Tensor,
        partial_selection: torch.Tensor,
        transformed_state: torch.Tensor,
        embedded_interaction: torch.Tensor,
        card_indices: torch.Tensor,
        include_stop_token: bool,
    ) -> torch.Tensor:
        return self._service.submit_target_score(
            _to_cpu_if_needed(candidates),
            _to_cpu_if_needed(partial_selection),
            _to_cpu_if_needed(transformed_state),
            _to_cpu_if_needed(embedded_interaction),
            _to_cpu_if_needed(card_indices),
            include_stop_token,
        )

    def game_finished(self) -> None:
        self._service.game_finished()


class BatchedInferenceService:
    _SHUTDOWN_ERROR = RuntimeError("BatchedInferenceService is shut down")

    def __init__(
        self,
        network: KumpelNetwork,
        selector: Selector,
        compute_device: torch.device,
        max_batch: int = 8,
        linger_ms: float = 2.0,
    ):
        self.network = network
        self.selector = selector
        self.compute_device = compute_device
        self.max_batch = max_batch
        self.linger_s = linger_ms / 1000.0

        self._live_games = 0
        self._lock = threading.Lock()
        self._move_cond = threading.Condition(self._lock)
        self._target_cond = threading.Condition(self._lock)
        self._move_pending: list[_MoveEvalRequest] = []
        self._target_pending: list[_TargetScoreRequest] = []
        self._shutdown = False

        self._move_worker = threading.Thread(
            target=self._move_eval_worker, name="move-eval-worker", daemon=True
        )
        self._target_worker = threading.Thread(
            target=self._target_score_worker, name="target-score-worker", daemon=True
        )
        self._move_worker.start()
        self._target_worker.start()

    def register_game(self) -> None:
        with self._lock:
            self._live_games += 1

    def game_finished(self) -> None:
        with self._lock:
            self._live_games = max(0, self._live_games - 1)
            self._move_cond.notify_all()
            self._target_cond.notify_all()

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown = True
            move_pending = self._move_pending
            target_pending = self._target_pending
            self._move_pending = []
            self._target_pending = []
            self._move_cond.notify_all()
            self._target_cond.notify_all()

        for request in move_pending + target_pending:
            request.error = self._SHUTDOWN_ERROR
            request.event.set()

        self._move_worker.join(timeout=5.0)
        self._target_worker.join(timeout=5.0)
        if self._move_worker.is_alive() or self._target_worker.is_alive():
            raise RuntimeError(
                "BatchedInferenceService workers did not stop within 5 seconds"
            )

    def _should_flush(self, pending_count: int, first_submit_time: float | None) -> bool:
        if pending_count == 0:
            return False
        if pending_count >= self.max_batch:
            return True
        if pending_count >= self._live_games:
            return True
        if first_submit_time is not None:
            if time.monotonic() - first_submit_time >= self.linger_s:
                return True
        return False

    def submit_move_eval(
        self,
        state_bytes: bytes,
        interactions_bytes: list[bytes],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        request = _MoveEvalRequest(
            state_bytes=state_bytes,
            interactions_bytes=interactions_bytes,
        )
        with self._lock:
            if self._shutdown:
                raise self._SHUTDOWN_ERROR
            self._move_pending.append(request)
            self._move_cond.notify()
        request.event.wait()
        if request.error is not None:
            raise request.error
        return request.result

    def submit_target_score(
        self,
        candidates: torch.Tensor,
        partial_selection: torch.Tensor,
        transformed_state: torch.Tensor,
        embedded_interaction: torch.Tensor,
        card_indices: torch.Tensor,
        include_stop_token: bool,
    ) -> torch.Tensor:
        request = _TargetScoreRequest(
            candidates=candidates,
            partial_selection=partial_selection,
            transformed_state=transformed_state,
            embedded_interaction=embedded_interaction,
            card_indices=card_indices,
            include_stop_token=include_stop_token,
        )
        with self._lock:
            if self._shutdown:
                raise self._SHUTDOWN_ERROR
            self._target_pending.append(request)
            self._target_cond.notify()
        request.event.wait()
        if request.error is not None:
            raise request.error
        return request.result

    def _move_eval_worker(self) -> None:
        pending: list[_MoveEvalRequest] = []
        while True:
            with self._lock:
                if self._shutdown and not pending and not self._move_pending:
                    return
                if not pending and self._move_pending:
                    pending = self._move_pending
                    self._move_pending = []
                elif not pending:
                    self._move_cond.wait(timeout=0.1)
                    continue

                first_submit_time = time.monotonic()
                while not self._should_flush(len(pending), first_submit_time):
                    if self._shutdown:
                        break
                    self._move_cond.wait(timeout=0.05)
                    if self._move_pending:
                        pending.extend(self._move_pending)
                        self._move_pending = []
                        if self._should_flush(len(pending), first_submit_time):
                            break

                batch = pending
                pending = []

            if not batch:
                continue

            try:
                with torch.inference_mode():
                    results = self._run_move_eval_batch(batch)
                for request, result in zip(batch, results):
                    request.result = result
                    request.event.set()
            except BaseException as exc:
                for request in batch:
                    request.error = exc
                    request.event.set()

    def _target_score_worker(self) -> None:
        pending: list[_TargetScoreRequest] = []
        while True:
            with self._lock:
                if self._shutdown and not pending and not self._target_pending:
                    return
                if not pending and self._target_pending:
                    pending = self._target_pending
                    self._target_pending = []
                elif not pending:
                    self._target_cond.wait(timeout=0.1)
                    continue

                first_submit_time = time.monotonic()
                while not self._should_flush(len(pending), first_submit_time):
                    if self._shutdown:
                        break
                    self._target_cond.wait(timeout=0.05)
                    if self._target_pending:
                        pending.extend(self._target_pending)
                        self._target_pending = []
                        if self._should_flush(len(pending), first_submit_time):
                            break

                batch = pending
                pending = []

            if not batch:
                continue

            try:
                with torch.inference_mode():
                    results = self._run_target_score_batch(batch)
                for request, result in zip(batch, results):
                    request.result = result
                    request.event.set()
            except BaseException as exc:
                for request in batch:
                    request.error = exc
                    request.event.set()

    def _run_move_eval_batch(
        self, batch: list[_MoveEvalRequest]
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        game_states = [r.state_bytes for r in batch]
        interactions_per_game = [r.interactions_bytes for r in batch]

        (
            scores,
            transformed_state,
            embedded_interactions,
            card_indices,
            int_mask,
            state_mask,
        ) = self.network.forward_batch(game_states, interactions_per_game)

        results: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for i in range(len(batch)):
            valid_state = state_mask[i]
            valid_int = int_mask[i]
            results.append(
                (
                    scores[i, valid_int],
                    transformed_state[i, valid_state],
                    embedded_interactions[i, valid_int],
                    card_indices[i],
                )
            )
        return results

    def _run_target_score_batch(
        self, batch: list[_TargetScoreRequest]
    ) -> list[torch.Tensor]:
        device = self.compute_device
        candidates_per_game = [r.candidates.to(device) for r in batch]
        partial_per_game = [r.partial_selection.to(device) for r in batch]
        state_per_game = [r.transformed_state.to(device) for r in batch]
        interaction_per_game = [r.embedded_interaction.to(device) for r in batch]
        card_indices_per_game = [r.card_indices.to(device) for r in batch]
        include_stop = [r.include_stop_token for r in batch]

        batched_scores = self.selector.forward_batch(
            candidates_per_game,
            partial_per_game,
            state_per_game,
            interaction_per_game,
            card_indices_per_game,
            include_stop,
        )

        results: list[torch.Tensor] = []
        for i, request in enumerate(batch):
            n_candidates = request.candidates.size(0)
            n_scores = n_candidates + (1 if request.include_stop_token else 0)
            results.append(batched_scores[i, :n_scores])
        return results
