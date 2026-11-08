"""103 immutable token-ID workloads shared by independently executed engines."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def prompts(model,batch):
    path=ROOT/'benchmarks/results/step090/numerics_contract.json'
    source=json.loads(path.read_text())['models'][model]['prompt_token_ids']
    lengths=[128] if batch==1 else [128,120,112,104]
    # Rotations give distinct histories, expose incorrect batch-row association.
    return [[source[(i+19*j)%len(source)] for i in range(n)] for j,n in enumerate(lengths)]
