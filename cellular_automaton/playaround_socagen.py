import torch

from cellular_automaton.soca_gen import (
    FORWARD,
    REVERSE,
    MaterializedSOCADataset,
    SOCADataset,
    generate_soca_batch,
    rollout_soca,
    soca_forward,
    soca_reverse,
    soca_step,
)


def split_state(state_sequence):
    """Split [..., N] into previous and current [..., N/2] rows."""
    midpoint = state_sequence.shape[-1] // 2
    return (
        state_sequence[..., :midpoint],
        state_sequence[..., midpoint:],
    )


def convert_to_four_token_ids(state_sequence):
    """
    Previous: bit 0 -> token 1, bit 1 -> token 2
    Current:  bit 0 -> token 3, bit 1 -> token 4
    """
    previous, current = split_state(state_sequence)

    previous_token_ids = previous + 1
    current_token_ids = current + 3

    return torch.cat(
        (previous_token_ids, current_token_ids),
        dim=-1,
    )


def print_state(name, state_sequence):
    previous, current = split_state(state_sequence)

    print(f"\n{name}")
    print("complete state:", state_sequence)
    print("complete shape:", tuple(state_sequence.shape))
    print("previous row:", previous)
    print("previous shape:", tuple(previous.shape))
    print("current row:", current)
    print("current shape:", tuple(current.shape))


def inspect_generated_batch():
    print("\n========== GENERATED BATCH ==========")

    batch_size = 2
    sequence_length = 8
    num_repeats = 4

    # Independent reproducible random streams for the two initial rows.
    previous_generator = torch.Generator().manual_seed(100)
    current_generator = torch.Generator().manual_seed(200)

    # Example 0 reverses on repeat 3: F F R R
    # Example 1 never reverses:          F F F F
    first_reverse_repeat = torch.tensor([3, 5])

    batch = generate_soca_batch(
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_repeats=num_repeats,
        first_reverse_repeat=first_reverse_repeat,
        previous_generator=previous_generator,
        current_generator=current_generator,
    )

    states = batch["state_sequence"]
    schedules = batch["direction_schedule"]

    print("Batch fields:", list(batch))
    print("state_sequence shape:", tuple(states.shape))
    print(f"state seq itself : {states}")
    print("direction_schedule shape:", tuple(schedules.shape))
    print("direction schedule = ", schedules)
    print("first_reverse_repeat:", batch["first_reverse_repeat"])
    print("example_id:", batch["example_id"])

    print("\nDirection encoding:")
    print("FORWARD =", FORWARD)
    print("REVERSE =", REVERSE)
    print("schedules:")
    print(schedules)

    for example_index in range(batch_size):
        print_state(
            f"Initial state for example {example_index}",
            states[example_index],
        )

        token_ids = convert_to_four_token_ids(states[example_index])
        print("four-token-ID representation:", token_ids)
        print("token-ID shape:", tuple(token_ids.shape))

    trajectory = rollout_soca(states, schedules)
    print(f"======TRAJECTORY=======")
    print(f"======TRAJECTORY=======")
    print(f"======TRAJECTORY=======")
    print("\nTrajectory shape:", tuple(trajectory.shape))
    print("Expected shape: [batch_size, num_repeats, sequence_length]")
    print(
        "Expected values:",
        (batch_size, num_repeats, sequence_length),
    )

    for example_index in range(batch_size):
        print(f"\n--- Example {example_index} trajectory ---")
        print("schedule:", schedules[example_index])

        print_state("Initial state", states[example_index])

        for repeat_index in range(num_repeats):
            direction = schedules[example_index, repeat_index]
            direction_name = "REVERSE" if direction.item() == REVERSE else "FORWARD"

            print_state(
                f"State after repeat {repeat_index + 1} ({direction_name})",
                trajectory[example_index, repeat_index],
            )


def inspect_individual_transitions():
    print("\n========== INDIVIDUAL TRANSITIONS ==========")

    # N=8, so each row has length 4.
    state = torch.tensor(
        [0, 1, 1, 0, 1, 0, 0, 1],
        dtype=torch.long,
    )

    print_state("Original state", state)

    forward_state = soca_forward(state)
    print_state("After one forward transition", forward_state)

    restored_state = soca_reverse(forward_state)
    print_state("After reversing the forward transition", restored_state)

    print(
        "Reverse restored the original:",
        torch.equal(restored_state, state),
    )

    selected_forward = soca_step(state, FORWARD)
    selected_reverse = soca_step(state, REVERSE)

    print_state("soca_step with FORWARD", selected_forward)
    print_state("soca_step with REVERSE", selected_reverse)


def inspect_indexed_dataset():
    print("\n========== INDEXED DATASET ==========")

    dataset = SOCADataset(
        num_samples=5,
        sequence_length=8,
        num_repeats=4,
        bernoulli_p=0.5,
        seed=123,
        schedule_seed=456,
        first_reverse_repeat=3,
    )

    print("Dataset length:", len(dataset))

    item = dataset[0]

    print("Item fields:", list(item))
    print("state_sequence shape:", tuple(item["state_sequence"].shape))
    print("direction_schedule shape:", tuple(item["direction_schedule"].shape))
    print("first_reverse_repeat:", item["first_reverse_repeat"])
    print("example_id:", item["example_id"])

    print_state("Indexed dataset item 0", item["state_sequence"])

    # The same seed and index should recreate exactly the same item.
    item_again = dataset[0]

    print(
        "Requesting index 0 twice gives the same state:",
        torch.equal(
            item["state_sequence"],
            item_again["state_sequence"],
        ),
    )


def inspect_materialized_dataset():
    print("\n========== MATERIALIZED DATASET ==========")

    dataset = MaterializedSOCADataset(
        num_samples=5,
        sequence_length=8,
        num_repeats=4,
        bernoulli_p=0.5,
        seed=123,
        schedule_seed=456,

        # For R=4 there are five categories:
        # 1 -> R R R R
        # 2 -> F R R R
        # 3 -> F F R R
        # 4 -> F F F R
        # 5 -> F F F F
        #
        # This enables only categories 3 and 5.
        reversal_weights=[0, 0, 1, 0, 1],
    )

    print("Dataset length:", len(dataset))
    print("All state sequences shape:", tuple(dataset.state_sequences.shape))
    print(
        "All direction schedules shape:",
        tuple(dataset.direction_schedules.shape),
    )
    print("Storage bytes:", dataset.storage_bytes)

    for index in range(len(dataset)):
        item = dataset[index]

        print(f"\nMaterialized item {index}")
        print("state_sequence:", item["state_sequence"])
        print("schedule:", item["direction_schedule"])
        print("first reverse repeat:", item["first_reverse_repeat"])


if __name__ == "__main__":
    inspect_generated_batch()
    inspect_individual_transitions()
    inspect_indexed_dataset()
    inspect_materialized_dataset()