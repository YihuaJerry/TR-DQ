# Repository-local import bootstrap (keeps the original scripts directly executable).
from pathlib import Path as _Path
import sys as _sys
_SCRIPT_FILE = _Path(__file__).resolve()
_REPO_ROOT = _SCRIPT_FILE.parents[3]
_BACKEND_ROOT = _SCRIPT_FILE.parents[1]
for _local_path in (_REPO_ROOT, _BACKEND_ROOT):
    if str(_local_path) not in _sys.path:
        _sys.path.insert(0, str(_local_path))
import argparse

import torch

def split_qkv(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'qkv' in key:
            prefix, suffix = key.split('.qkv.')
            q_key = prefix + '.q.' + suffix
            k_key = prefix + '.k.' + suffix
            v_key = prefix + '.v.' + suffix
            print(q_key,k_key,v_key)
            new_state_dict[q_key] = value[:value.size(0) // 3]
            new_state_dict[k_key] = value[value.size(0) // 3: 2 * (value.size(0) // 3)]
            new_state_dict[v_key] = value[2 * (value.size(0) // 3):]
        else:
            new_state_dict[key] = value
    return new_state_dict

def main():
    parser = argparse.ArgumentParser(description="Split fused OpenSora QKV tensors")
    parser.add_argument("input", help="Original OpenSora checkpoint")
    parser.add_argument("output", help="Destination checkpoint")
    args = parser.parse_args()

    state_dict = torch.load(args.input, map_location="cpu")
    new_state_dict = split_qkv(state_dict)
    output = _Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_state_dict, output)


if __name__ == "__main__":
    main()
