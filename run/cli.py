from __future__ import annotations
import argparse
from main import ONIMonolith, MonolithConfig

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("prompt")
    p.add_argument("--task", default="chat")
    args = p.parse_args()
    oni = ONIMonolith(MonolithConfig())
    if args.task == "chat":
        msgs = [{"role":"user","content": args.prompt}]
        print(oni.chat(msgs))
    elif args.task == "code":
        print(oni.code(args.prompt))
    else:
        print("Unsupported in demo CLI")
