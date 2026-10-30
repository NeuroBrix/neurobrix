"""
Tensor Parallel Block Sharding

Creates and manages weight shards for tensor parallel execution.

Key features:
- LAZY: Creates shards on first runtime, not import
- HASH-BASED: Only re-shards if source weights changed
- ZERO HARDCODE: Shard axis detected from tensor shapes
- CACHED: Shards stored in ~/.neurobrix/cache/<name>/components/<comp>/block_shards/

Cache structure:
  ~/.neurobrix/cache/<model>/components/<component>/
  ├── weights/                    # Original merged weights
  │   └── model.safetensors
  └── block_shards/              # TP shards (created lazily)
      ├── manifest.json          # Shard metadata + source hash
      ├── block.0/
      │   ├── shard_0.safetensors
      │   └── shard_1.safetensors
      └── block.N/
          ├── shard_0.safetensors
          └── shard_1.safetensors
"""

import json
import hashlib
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Use consolidated parallel loader (eliminates code duplication)
from neurobrix.core.io.loader import load_files_parallel, DEFAULT_NUM_WORKERS

# Parallel loading workers for TP sharding (matches system.yml io.num_workers)
TP_SHARD_WORKERS = DEFAULT_NUM_WORKERS


