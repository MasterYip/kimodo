"""LocoEditor v2 — Interactive Viser web GUI for locomotion motion generation.

Three tabs: Config (YAML-equivalent widgets), Generate (batch with progress),
Visualize (3D robot playback with camera presets).

Usage:
    cd /data/masteryip/kimodo/kimodo
    source scripts/env.sh
    PYTHONPATH=. python3 -m locomotion_framework.editor \\
        --model kimodo-g1-rp --port 7861

Access:
    ssh -L 7861:127.0.0.1:7861 user@<server-ip> -p 22222 -N
    Open http://127.0.0.1:7861
"""

import argparse
import sys

from .app import LocoEditor


def main():
    parser = argparse.ArgumentParser(
        description="Loco Editor — Interactive locomotion config editor with 3D viz"
    )
    parser.add_argument(
        "--model", type=str, default="kimodo-g1-rp",
        help="Kimodo model name (default: kimodo-g1-rp)"
    )
    parser.add_argument(
        "--port", type=int, default=7861,
        help="Viser server port (default: 7861)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Optional initial YAML config path"
    )
    args = parser.parse_args()

    editor = LocoEditor(
        model_name=args.model,
        port=args.port,
        config_path=args.config,
    )
    editor.run()


if __name__ == "__main__":
    main()
