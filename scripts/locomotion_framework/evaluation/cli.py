"""CLI for reusable locomotion output evaluation."""
import argparse
from .core import evaluate_output_root

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("input_root")
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--per-motion-plots",action="store_true")
    parser.add_argument("--comparison", action="append", default=[])
    args=parser.parse_args()
    rows=evaluate_output_root(args.input_root,args.output_dir,args.per_motion_plots,args.comparison)
    print(f"PASS: evaluated {len(rows)} motions -> {args.output_dir}")

if __name__ == "__main__": main()
