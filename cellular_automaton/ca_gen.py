"""Rule 30 transitions and reproducible Bernoulli datasets."""

import torch
from torch.utils.data import Dataset


def rule30(states):
    """Apply one periodic-boundary Rule 30 update along the final dimension.

    ``states`` may be a single row with shape ``[num_cells]`` or a batch with
    shape ``[..., num_cells]``.  Values are expected to be binary integers.
    """
    if states.ndim == 0 or states.shape[-1] == 0:
        raise ValueError("Rule 30 requires at least one cell.")

    left = torch.roll(states, shifts=1, dims=-1)
    right = torch.roll(states, shifts=-1, dims=-1)
    return left ^ (states | right)


def apply_rule30(states, steps=1):
    """Apply Rule 30 ``steps`` times, returning the final state."""
    if steps <= 0:
        raise ValueError("steps must be positive.")

    result = states
    for _ in range(steps):
        result = rule30(result)
    return result


def generate_rule30_batch(
    batch_size,
    num_cells,
    device="cpu",
    *,
    steps=1,
    bernoulli_p=0.5,
    generator=None,
):
    """Generate a Bernoulli batch and its Rule 30 target rows."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_cells <= 0:
        raise ValueError("num_cells must be positive.")
    if not 0.0 <= bernoulli_p <= 1.0:
        raise ValueError("bernoulli_p must lie in [0, 1].")

    initial_state = (
        torch.rand(
            (batch_size, num_cells),
            device=device,
            generator=generator,
        )
        < bernoulli_p
    ).long()
    next_state = apply_rule30(initial_state, steps=steps)
    return {
        "input_id": initial_state,
        "label": next_state,
    }


class Rule30Dataset(Dataset):
    """Finite, index-reproducible samples from a Bernoulli Rule 30 task.

    The dataset is random in distribution but deterministic by index: the same
    ``seed`` and sample index always produce the same Bernoulli row.  This makes
    fixed validation and exact experiment reproduction possible while a
    DataLoader remains free to shuffle the sample order.
    """

    def __init__(
        self,
        *,
        num_samples,
        num_cells,
        steps=1,
        bernoulli_p=0.5,
        seed=0,
    ):
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if num_cells <= 0:
            raise ValueError("num_cells must be positive.")
        if steps <= 0:
            raise ValueError("steps must be positive.")
        if not 0.0 <= bernoulli_p <= 1.0:
            raise ValueError("bernoulli_p must lie in [0, 1].")

        self.num_samples = int(num_samples)
        self.num_cells = int(num_cells)
        self.steps = int(steps)
        self.bernoulli_p = float(bernoulli_p)
        self.seed = int(seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if not 0 <= index < self.num_samples:
            raise IndexError(index)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + int(index))
        initial_state = (
            torch.rand(self.num_cells, generator=generator) < self.bernoulli_p
        ).long()
        target_state = apply_rule30(initial_state, steps=self.steps)

        return {
            "input_id": initial_state,
            "label": target_state,
        }


class MaterializedRule30Dataset(Dataset):
    """A fixed Rule 30 split generated once and retained in CPU memory.

    Rows are stored as uint8 to keep the fixed dataset compact.  The trainer
    and evaluators convert complete batches to integer token tensors when they
    move them to the model device.
    """

    def __init__(
        self,
        *,
        num_samples,
        num_cells,
        steps=1,
        bernoulli_p=0.5,
        seed=0,
    ):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        batch = generate_rule30_batch(
            batch_size=num_samples,
            num_cells=num_cells,
            steps=steps,
            bernoulli_p=bernoulli_p,
            generator=generator,
        )
        self.inputs = batch["input_id"].to(torch.uint8)
        self.labels = batch["label"].to(torch.uint8)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        return {
            "input_id": self.inputs[index],
            "label": self.labels[index],
        }

    @property
    def storage_bytes(self):
        return (
            self.inputs.numel() * self.inputs.element_size()
            + self.labels.numel() * self.labels.element_size()
        )
