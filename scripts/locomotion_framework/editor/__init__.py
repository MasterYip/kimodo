"""Loco Editor — Viser-based web GUI for the Kimodo locomotion framework.

Provides interactive config editing, batch generation with progress tracking,
and 3D robot visualisation in a single browser tab.

Usage:
    cd /data/masteryip/kimodo/kimodo
    source scripts/env.sh
    PYTHONPATH=. python3 -m locomotion_framework.editor [--model MODEL] [--port PORT]
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Loco Editor — Viser GUI for locomotion motion generation"
    )
    parser.add_argument(
        "--model", type=str, default="kimodo-g1-rp",
        help="Model to load (default: kimodo-g1-rp)",
    )
    parser.add_argument(
        "--port", type=int, default=7861,
        help="Port for the viser web server (default: 7861)",
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Path to initial YAML config file on the server",
    )

    args = parser.parse_args()

    from .app import LocoEditor

    editor = LocoEditor(
        model_name=args.model,
        port=args.port,
        config_path=args.config,
    )
    editor.run()


if __name__ == "__main__":
    main()
