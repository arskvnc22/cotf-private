"""Reversible second-order Rule 30 transitions and reproducible datasets.

A public SOCA state is one complete sequence whose first half is the previous
row and whose second half is the current row. The module owns exact dynamics
and compact raw binary data, but not token IDs, supervision, or loss policy.
"""

import torch
from torch.utils.data import Dataset

from .ca_gen import rule30


FORWARD = 0
REVERSE = 1
_MAX_TORCH_SEED = 2**63 - 1
debug = False
def printifdeb(statement):
    if debug == True:
        print(statement)


def _positive_int(value, name):
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _sequence_length(value):
    value = _positive_int(value, "sequence_length")
    if value % 2 != 0:
        raise ValueError(
            "sequence_length must be even so it can contain equal-length "
            "previous and current rows."
        )
    return value


def _validate_binary_tensor(tensor, name):
    if tensor.dtype.is_floating_point or tensor.dtype.is_complex:
        raise TypeError(f"{name} must use an integer or boolean dtype.")
    if not bool(torch.all((tensor == 0) | (tensor == 1)).item()):
        raise ValueError(f"{name} must contain only binary values 0 and 1.")


def _validate_state_sequence(state_sequence):
    if not isinstance(state_sequence, torch.Tensor):
        raise TypeError("state_sequence must be a torch.Tensor.")
    if state_sequence.ndim < 1:
        raise ValueError("state_sequence must have shape [..., sequence_length].")
    _sequence_length(state_sequence.shape[-1])
    _validate_binary_tensor(state_sequence, "state_sequence")


def _split_state_sequence(state_sequence):
    row_length = state_sequence.shape[-1] // 2
    return state_sequence[..., :row_length], state_sequence[..., row_length:]


def _as_direction_tensor(direction, device):
    if isinstance(direction, torch.Tensor):
        direction = direction.to(device=device)
    else:
        direction = torch.as_tensor(direction, device=device)
    _validate_binary_tensor(direction, "direction")
    return direction


def _soca_forward_unchecked(state_sequence):
    previous, current = _split_state_sequence(state_sequence)
    next_current = previous ^ rule30(current)
    return torch.cat((current, next_current), dim=-1)


def _soca_reverse_unchecked(state_sequence):
    previous, current = _split_state_sequence(state_sequence)
    prior_previous = current ^ rule30(previous)
    return torch.cat((prior_previous, previous), dim=-1)


def soca_forward(state_sequence):
    """Apply ``(A, B) -> (B, A XOR rule30(B))`` to concatenated rows."""
    _validate_state_sequence(state_sequence)
    return _soca_forward_unchecked(state_sequence)


def soca_reverse(state_sequence):
    """Apply ``(A, B) -> (B XOR rule30(A), A)`` to concatenated rows."""
    _validate_state_sequence(state_sequence)
    return _soca_reverse_unchecked(state_sequence)


def _soca_step_unchecked(state_sequence, direction):
    forward_state = _soca_forward_unchecked(state_sequence)
    reverse_state = _soca_reverse_unchecked(state_sequence)
    reverse_mask = direction.bool()
    if direction.ndim != 0:
        reverse_mask = reverse_mask[..., None]
    return torch.where(reverse_mask, reverse_state, forward_state)


def soca_step(state_sequence, direction):
    """Apply a forward or reverse transition independently per example."""
    _validate_state_sequence(state_sequence)
    direction = _as_direction_tensor(direction, state_sequence.device)
    leading_shape = state_sequence.shape[:-1]
    if direction.ndim != 0 and tuple(direction.shape) != tuple(leading_shape):
        raise ValueError(
            "direction must be scalar or match the leading state_sequence "
            f"shape {tuple(leading_shape)}; got {tuple(direction.shape)}."
        )
    return _soca_step_unchecked(state_sequence, direction)


def _first_reverse_tensor(first_reverse_repeat, *, num_repeats, device):
    no_reversal = num_repeats + 1
    if first_reverse_repeat is None:
        return torch.tensor(no_reversal, device=device, dtype=torch.long)
    values = torch.as_tensor(first_reverse_repeat, device=device, dtype=torch.long)
    if values.ndim > 1:
        raise ValueError("first_reverse_repeat must be a scalar or one-dimensional.")
    if not bool(torch.all((values >= 1) & (values <= no_reversal)).item()):
        raise ValueError(
            "first_reverse_repeat must lie in [1, num_repeats + 1], where "
            "num_repeats + 1 means no reversal within the rollout."
        )
    return values


def make_single_switch_schedule(
    num_repeats,
    first_reverse_repeat=None,
    *,
    batch_size=None,
    device="cpu",
):
    """Build persistent forward-then-reverse direction schedules."""
    num_repeats = _positive_int(num_repeats, "num_repeats")
    if batch_size is not None:
        batch_size = _positive_int(batch_size, "batch_size")

    first_reverse = _first_reverse_tensor(
        first_reverse_repeat,
        num_repeats=num_repeats,
        device=device,
    )
    if first_reverse.ndim == 0:
        if batch_size is not None:
            first_reverse = first_reverse.expand(batch_size)
    else:
        if batch_size is not None and first_reverse.shape[0] != batch_size:
            raise ValueError(
                "first_reverse_repeat length must equal batch_size; got "
                f"{first_reverse.shape[0]} and {batch_size}."
            )
        if batch_size is None:
            batch_size = int(first_reverse.shape[0])

    repeats = torch.arange(1, num_repeats + 1, device=device)
    if first_reverse.ndim == 0:
        return (repeats >= first_reverse).to(torch.uint8)
    return (repeats[None, :] >= first_reverse[:, None]).to(torch.uint8)


def _validate_direction_schedule(direction_schedule, state_sequence):
    schedule = _as_direction_tensor(direction_schedule, state_sequence.device)
    if schedule.ndim not in (1, 2):
        raise ValueError("direction_schedule must have shape [R] or [B, R].")
    if schedule.shape[-1] == 0:
        raise ValueError("direction_schedule requires at least one repeat.")

    if state_sequence.ndim == 1:
        if schedule.ndim != 1:
            raise ValueError("An unbatched state_sequence requires a [R] schedule.")
    elif state_sequence.ndim == 2:
        if schedule.ndim == 2 and schedule.shape[0] != state_sequence.shape[0]:
            raise ValueError(
                "Batched direction_schedule and state_sequence batch sizes differ."
            )
    else:
        raise ValueError(
            "rollout_soca supports unbatched or singly batched state sequences."
        )
    return schedule.to(torch.uint8)


def rollout_soca(state_sequence, direction_schedule):
    """Return the complete concatenated SOCA state after every transition.

    ``[B, N]`` states and ``[B, R]`` schedules produce ``[B, R, N]``
    trajectories. A common ``[R]`` schedule may be broadcast over a batch.
    The initial state is not included in the returned trajectory.
    """
    _validate_state_sequence(state_sequence)
    schedule = _validate_direction_schedule(direction_schedule, state_sequence)

    current = state_sequence
    states = []
    for repeat_index in range(schedule.shape[-1]):
        direction = (
            schedule[repeat_index]
            if schedule.ndim == 1
            else schedule[:, repeat_index]
        )
        current = _soca_step_unchecked(current, direction)
        states.append(current)
    return torch.stack(states, dim=-2)


def _validate_reversal_weights(reversal_weights, num_repeats, device):
    weights = torch.as_tensor(reversal_weights, device=device, dtype=torch.float64)
    if weights.ndim != 1 or weights.numel() != num_repeats + 1:
        raise ValueError(
            "reversal_weights must contain one weight for each first reverse "
            "repeat 1 .. num_repeats + 1."
        )
    if not bool(torch.all(torch.isfinite(weights)).item()):
        raise ValueError("reversal_weights must be finite.")
    if not bool(torch.all(weights >= 0).item()) or float(weights.sum().item()) <= 0:
        raise ValueError("reversal_weights must be non-negative with a positive sum.")
    return weights


def sample_first_reverse_repeats(
    batch_size,
    num_repeats,
    reversal_weights,
    *,
    generator=None,
    device="cpu",
):
    """Sample one-based first reverse repeats from explicit category weights."""
    batch_size = _positive_int(batch_size, "batch_size")
    num_repeats = _positive_int(num_repeats, "num_repeats")
    weights = _validate_reversal_weights(reversal_weights, num_repeats, device)
    categories = torch.multinomial(
        weights,
        batch_size,
        replacement=True,
        generator=generator,
    )
    return categories.to(torch.long) + 1


def _first_reverse_from_schedule(schedule):
    num_repeats = schedule.shape[-1]
    reverse = schedule.bool()
    positions = torch.arange(1, num_repeats + 1, device=schedule.device)
    positions = positions.expand_as(schedule)
    no_reversal = torch.full_like(positions, num_repeats + 1)
    return torch.where(reverse, positions, no_reversal).amin(dim=-1)


def generate_soca_batch(
    batch_size,
    sequence_length,
    device="cpu",
    *,
    num_repeats,
    bernoulli_p=0.5,
    first_reverse_repeat=None,
    reversal_weights=None,
    direction_schedule=None,
    initial_state_sequence=None,
    previous_generator=None,
    current_generator=None,
    schedule_generator=None,
):
    """Generate compact raw ``[B, N]`` SOCA states and direction schedules.

    The first ``N/2`` bits are the previous row and the remaining bits are the
    current row. Token IDs and exact trajectory states are deliberately not
    materialized here.
    """
    batch_size = _positive_int(batch_size, "batch_size")
    sequence_length = _sequence_length(sequence_length)
    num_repeats = _positive_int(num_repeats, "num_repeats")
    if not 0.0 <= bernoulli_p <= 1.0:
        raise ValueError("bernoulli_p must lie in [0, 1].")

    schedule_policies = sum(
        value is not None
        for value in (first_reverse_repeat, reversal_weights, direction_schedule)
    )
    if schedule_policies > 1:
        raise ValueError(
            "Provide only one of first_reverse_repeat, reversal_weights, or "
            "direction_schedule."
        )

    if initial_state_sequence is None:
        row_length = sequence_length // 2
        previous = (
            torch.rand(
                (batch_size, row_length),
                device=device,
                generator=previous_generator,
            )
            < bernoulli_p
        )
        current = (
            torch.rand(
                (batch_size, row_length),
                device=device,
                generator=current_generator,
            )
            < bernoulli_p
        )
        state_sequence = torch.cat((previous, current), dim=-1).long()
        printifdeb("===========inside gen==============")
        printifdeb("===========inside gen==============")
        printifdeb(f"args {batch_size,num_repeats,bernoulli_p,first_reverse_repeat,reversal_weights,direction_schedule,initial_state_sequence,previous_generator,current_generator,schedule_generator}")
        printifdeb(f"state sequence from isnide batch is of shape {state_sequence.shape}")
        printifdeb(f"it looks like {state_sequence}")
        printifdeb("===========inside gen==============")
        printifdeb("===========inside gen==============")
        printifdeb("===========inside gen==============")
        
    else:
        if not isinstance(initial_state_sequence, torch.Tensor):
            raise TypeError("initial_state_sequence must be a torch.Tensor.")
        state_sequence = initial_state_sequence.to(device=device)
        _validate_state_sequence(state_sequence)
        if tuple(state_sequence.shape) != (batch_size, sequence_length):
            raise ValueError(
                "initial_state_sequence must have shape "
                f"({batch_size}, {sequence_length})."
            )

    if direction_schedule is not None:
        schedule = _as_direction_tensor(direction_schedule, torch.device(device))
        if schedule.ndim == 1:
            if schedule.shape[0] != num_repeats:
                raise ValueError("direction_schedule length must equal num_repeats.")
            schedule = schedule[None, :].expand(batch_size, -1)
        elif tuple(schedule.shape) != (batch_size, num_repeats):
            raise ValueError(
                "direction_schedule must have shape [R] or [batch_size, R]."
            )
        schedule = schedule.to(torch.uint8)
        first_reverse = _first_reverse_from_schedule(schedule)
    else:
        if reversal_weights is not None:
            first_reverse = sample_first_reverse_repeats(
                batch_size,
                num_repeats,
                reversal_weights,
                generator=schedule_generator,
                device=device,
            )
        else:
            first_reverse = _first_reverse_tensor(
                first_reverse_repeat,
                num_repeats=num_repeats,
                device=device,
            )
            if first_reverse.ndim == 0:
                first_reverse = first_reverse.expand(batch_size)
            elif first_reverse.shape[0] != batch_size:
                raise ValueError(
                    "first_reverse_repeat length must equal batch_size."
                )
        schedule = make_single_switch_schedule(
            num_repeats,
            first_reverse,
            batch_size=batch_size,
            device=device,
        )

    return {
        "state_sequence": state_sequence,
        "direction_schedule": schedule,
        "first_reverse_repeat": first_reverse.to(torch.long),
        "example_id": torch.arange(batch_size, device=device, dtype=torch.long),
    }


def _derived_seed(base_seed, *, index=0, stream=0):
    """Derive stable, separate per-index random-number substreams."""
    value = int(base_seed) % _MAX_TORCH_SEED
    value = value * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407
    value += int(index) * 3_202_034_522_624_059_733
    value += int(stream) * 393_555_900_037_000_384
    return value % _MAX_TORCH_SEED


class SOCADataset(Dataset):
    """Finite SOCA samples that are deterministic by seed and index."""

    def __init__(
        self,
        *,
        num_samples,
        sequence_length,
        num_repeats,
        bernoulli_p=0.5,
        seed=0,
        schedule_seed=None,
        first_reverse_repeat=None,
        reversal_weights=None,
    ):
        self.num_samples = _positive_int(num_samples, "num_samples")
        self.sequence_length = _sequence_length(sequence_length)
        self.row_length = self.sequence_length // 2
        self.num_repeats = _positive_int(num_repeats, "num_repeats")
        if not 0.0 <= bernoulli_p <= 1.0:
            raise ValueError("bernoulli_p must lie in [0, 1].")
        if first_reverse_repeat is not None and reversal_weights is not None:
            raise ValueError(
                "Provide either first_reverse_repeat or reversal_weights, not both."
            )

        self.bernoulli_p = float(bernoulli_p)
        self.seed = int(seed)
        self.schedule_seed = (
            _derived_seed(seed, stream=2)
            if schedule_seed is None
            else int(schedule_seed)
        )
        self.first_reverse_repeat = _first_reverse_tensor(
            first_reverse_repeat,
            num_repeats=self.num_repeats,
            device="cpu",
        )
        if self.first_reverse_repeat.ndim != 0:
            raise ValueError("Dataset first_reverse_repeat must be a scalar.")
        self.reversal_weights = (
            None
            if reversal_weights is None
            else _validate_reversal_weights(
                reversal_weights,
                self.num_repeats,
                "cpu",
            ).clone()
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if not 0 <= index < self.num_samples:
            raise IndexError(index)

        previous_generator = torch.Generator(device="cpu")
        current_generator = torch.Generator(device="cpu")
        schedule_generator = torch.Generator(device="cpu")
        previous_generator.manual_seed(
            _derived_seed(self.seed, index=index, stream=0)
        )
        current_generator.manual_seed(
            _derived_seed(self.seed, index=index, stream=1)
        )
        schedule_generator.manual_seed(
            _derived_seed(self.schedule_seed, index=index, stream=0)
        )

        previous = (
            torch.rand(self.row_length, generator=previous_generator)
            < self.bernoulli_p
        )
        current = (
            torch.rand(self.row_length, generator=current_generator)
            < self.bernoulli_p
        )
        state_sequence = torch.cat((previous, current), dim=-1).to(torch.uint8)

        if self.reversal_weights is None:
            first_reverse = self.first_reverse_repeat.clone()
        else:
            first_reverse = sample_first_reverse_repeats(
                1,
                self.num_repeats,
                self.reversal_weights,
                generator=schedule_generator,
            )[0]
        schedule = make_single_switch_schedule(
            self.num_repeats,
            first_reverse,
        )
        return {
            "state_sequence": state_sequence,
            "direction_schedule": schedule,
            "first_reverse_repeat": first_reverse.to(torch.long),
            "example_id": torch.tensor(index, dtype=torch.long),
        }


class MaterializedSOCADataset(Dataset):
    """A fixed compact SOCA split generated once and retained in CPU memory."""

    def __init__(
        self,
        *,
        num_samples,
        sequence_length,
        num_repeats,
        bernoulli_p=0.5,
        seed=0,
        schedule_seed=None,
        first_reverse_repeat=None,
        reversal_weights=None,
    ):
        num_samples = _positive_int(num_samples, "num_samples")
        schedule_seed = (
            _derived_seed(seed, stream=2)
            if schedule_seed is None
            else int(schedule_seed)
        )
        previous_generator = torch.Generator(device="cpu")
        current_generator = torch.Generator(device="cpu")
        schedule_generator = torch.Generator(device="cpu")
        previous_generator.manual_seed(_derived_seed(seed, stream=0))
        current_generator.manual_seed(_derived_seed(seed, stream=1))
        schedule_generator.manual_seed(_derived_seed(schedule_seed, stream=0))

        batch = generate_soca_batch(
            num_samples,
            sequence_length,
            num_repeats=num_repeats,
            bernoulli_p=bernoulli_p,
            first_reverse_repeat=first_reverse_repeat,
            reversal_weights=reversal_weights,
            previous_generator=previous_generator,
            current_generator=current_generator,
            schedule_generator=schedule_generator,
        )
        self.state_sequences = batch["state_sequence"].to(torch.uint8)
        self.direction_schedules = batch["direction_schedule"].to(torch.uint8)
        self.first_reverse_repeats = batch["first_reverse_repeat"].to(torch.long)
        self.example_ids = batch["example_id"].to(torch.long)

    def __len__(self):
        return self.state_sequences.shape[0]

    def __getitem__(self, index):
        return {
            "state_sequence": self.state_sequences[index],
            "direction_schedule": self.direction_schedules[index],
            "first_reverse_repeat": self.first_reverse_repeats[index],
            "example_id": self.example_ids[index],
        }

    @property
    def storage_bytes(self):
        tensors = (
            self.state_sequences,
            self.direction_schedules,
            self.first_reverse_repeats,
            self.example_ids,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)
