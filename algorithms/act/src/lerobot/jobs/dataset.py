# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Make a training dataset reachable from an HF Job pod.

The pod can't see the host's ~/.cache/huggingface/lerobot, so the dataset has to
live on the Hub: the pod downloads it by repo_id at train time (the forwarded
HF_TOKEN covers private datasets). A dataset already on the Hub is used as-is; a
local-only dataset is pushed to a PRIVATE repo first (never public).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

if TYPE_CHECKING:
    from huggingface_hub import HfApi


def ensure_dataset_available(
    repo_id: str,
    *,
    api: HfApi,
    root: str | Path | None = None,
    tags: list[str] | None = None,
) -> None:
    """Ensure repo_id resolves on the Hub, pushing a local-only dataset privately first.

    `root` selects an explicit local dataset tree; when omitted, the legacy
    `HF_LEROBOT_HOME / repo_id` location is used. `tags` are attached only when
    pushing (an already-on-Hub dataset is left untouched). Raises RuntimeError
    if the dataset is neither on the Hub nor at the selected local path.
    """
    if api.repo_exists(repo_id, repo_type="dataset"):
        return

    dataset_root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise RuntimeError(
            f"Dataset '{repo_id}' is not in the local cache at {info_path} and could not be "
            f"reached on the Hub — it may not exist, or be private and inaccessible with your "
            f"token. Record or download it first, or run `hf auth login`."
        )

    print(f"[dataset] '{repo_id}' is local-only; pushing to a PRIVATE Hub repo...")
    dataset = LeRobotDataset(repo_id, root=root) if root is not None else LeRobotDataset(repo_id)
    dataset.push_to_hub(private=True, tags=tags)
    print(f"[dataset] '{repo_id}' uploaded (private). The job will download it by repo_id.")
