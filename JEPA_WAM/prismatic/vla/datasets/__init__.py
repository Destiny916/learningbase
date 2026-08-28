"""Dataset exports.

The W1 LeRobot backend does not require the optional RLDS/dlimp stack.  Keep
this package importable in the lightweight Docker image used for statistics
and contract checks, while preserving the normal exports when dlimp exists.
"""

try:
    from .datasets import RLDSDataset, VLABatchTransform
except ModuleNotFoundError as exc:
    if exc.name != "dlimp":
        raise

    RLDSDataset = None  # type: ignore[assignment,misc]
    VLABatchTransform = None  # type: ignore[assignment,misc]
