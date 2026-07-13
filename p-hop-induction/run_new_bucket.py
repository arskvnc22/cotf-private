import json
from pathlib import Path

from phop_data import get_phop_task_spec, make_generator, write_phop_split

task = "phop_p4_seq32_a4_final_constructive"
spec = get_phop_task_spec(task)
print(f"spec : {spec}")

output_path = Path("data/p-hop") / task / "smoke.txt"

count = write_phop_split(
    output_path=output_path,
    spec=spec,
    num_examples=16,  # Must be divisible by hops=8
    seed=0,
    force=True,
    progress_every=1,
)

print(f"Wrote {count} examples to {output_path}")

generator = make_generator(spec, seed=0)
with output_path.open("r", encoding="utf-8") as handle:
    for example_idx, line in enumerate(handle, start=1):
        input_tokens, _ = json.loads(line)
        sequence = input_tokens[1:] if spec.include_hop_token else input_tokens
        answer, path = generator._compute_final_answer(sequence, spec.min_hops)
        answer_position = path[-1][0] if answer is not None else None
        print(
            f"Example {example_idx}: final answer={answer!r}, "
            f"position={answer_position}"
        )
