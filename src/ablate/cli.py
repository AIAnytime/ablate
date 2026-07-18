"""Command-line interface: ``ablate run|extract|generate|eval``."""
from __future__ import annotations

import argparse
import json
import sys

from .config import AblationConfig, RunConfig
from .pipeline import Ablator, run


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    p.add_argument("--device", default=None, help="cuda|mps|cpu (auto if unset)")
    p.add_argument("--dtype", default=None, help="float32|float16|bfloat16 (auto if unset)")
    p.add_argument("--trust-remote-code", action="store_true")


def cmd_run(args: argparse.Namespace) -> int:
    cfg = RunConfig(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        n_trials=args.trials,
        kl_weight=args.kl_weight,
        max_new_tokens=args.max_new_tokens,
        subspace=args.subspace,
        n_directions=args.n_directions,
        subspace_method=args.subspace_method,
        harmful_source=args.harmful_source,
        harmless_source=args.harmless_source,
        push_to_hub=args.push_to_hub,
        hf_token=args.hf_token,
        private=not args.public,
        output_dir=args.output,
        save_model=args.save_model,
    )
    result = run(cfg)
    print("\n=== BEST CONFIG ===")
    print(json.dumps(result.config.to_dict(), indent=2))
    print("=== METRICS ===")
    print(result.result)
    if result.result.samples:
        print("\n=== SAMPLE GENERATIONS (ablated, harmful eval prompts) ===")
        for s in result.result.samples[:5]:
            print("-", s.replace("\n", " ")[:200])
    print(f"\nSaved results to {cfg.output_dir}/result.json")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    abl = Ablator(args.model, device=args.device, dtype=args.dtype, trust_remote_code=args.trust_remote_code)
    dirs = abl.extract(method=args.method)
    abl.save_directions(args.output)
    print(f"Extracted directions {tuple(dirs.shape)} via '{args.method}' -> {args.output}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    abl = Ablator(args.model, device=args.device, dtype=args.dtype, trust_remote_code=args.trust_remote_code)
    abl.extract()
    # Default to the middle layer's direction, applied to all layers — the
    # standard single-direction abliteration setup.
    layer = args.layer if args.layer is not None else abl.lm.n_layers // 2
    cfg = AblationConfig(direction_layer=layer, alpha=args.alpha,
                         min_layer=args.min_layer, max_layer=args.max_layer or 10_000)
    baseline = abl.lm.generate([args.prompt], max_new_tokens=args.max_new_tokens)[0]
    ablated = abl.generate([args.prompt], config=cfg, max_new_tokens=args.max_new_tokens)[0]
    print("=== BASELINE ===\n" + baseline)
    print("\n=== ABLATED ===\n" + ablated)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from . import data
    from .harness import compare

    abl = Ablator(args.model, device=args.device, dtype=args.dtype, trust_remote_code=args.trust_remote_code)

    # Benchmark prompts.
    if args.benchmark == "builtin":
        prompts = data.split(data.load_builtin("harmful"), 0, args.n)[1][: args.n]
    elif args.benchmark == "harmbench":
        prompts = data.load_harmbench(n=args.n)
    elif args.benchmark == "advbench":
        prompts = data.load_advbench(n=args.n)
    elif args.benchmark == "jailbreakbench":
        prompts = data.load_jailbreakbench(n=args.n)
    else:
        prompts = data.load_hf(*args.benchmark.split(":")[1:3], n=args.n)

    # Find a config to evaluate (quick search on built-in eval data).
    if args.subspace:
        abl.extract_subspace(n_directions=args.n_directions)
        res = abl.search_subspace(n_trials=args.trials)
    else:
        abl.extract()
        res = abl.search(n_trials=args.trials)

    basis = abl._basis_for(res.config)
    out = compare(abl.lm, prompts, basis, res.config, judge=args.judge, max_new_tokens=args.max_new_tokens)
    print(json.dumps(out, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ablate", description="Directional ablation toolkit for language models.")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="full automated pipeline: extract -> optimize -> report")
    _add_common(r)
    r.add_argument("--trials", type=int, default=30)
    r.add_argument("--kl-weight", type=float, default=1.0)
    r.add_argument("--max-new-tokens", type=int, default=64)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--subspace", action="store_true", help="multi-direction (subspace) ablation")
    r.add_argument("--n-directions", type=int, default=4, help="subspace size (with --subspace)")
    r.add_argument("--subspace-method", default="band", choices=["band", "pca"])
    r.add_argument("--harmful-source", default=None,
                   help="None=built-in | advbench|harmbench|jailbreakbench | hf:<ds>:<col>[:<cfg>]")
    r.add_argument("--harmless-source", default=None,
                   help="None=built-in | alpaca | hf:<ds>:<col>[:<cfg>]")
    r.add_argument("--push-to-hub", default=None, metavar="REPO_ID",
                   help="bake + upload the model to this HF repo")
    r.add_argument("--hf-token", default=None, help="HF token (else uses HF_TOKEN env var)")
    r.add_argument("--public", action="store_true", help="make the pushed repo public (default: private)")
    r.add_argument("--output", default="ablate_out")
    r.add_argument("--save-model", action="store_true", help="bake best config and save checkpoint locally")
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("extract", help="extract and save candidate directions")
    _add_common(e)
    e.add_argument("--method", default="diff_of_means", choices=["diff_of_means", "pca", "probe"])
    e.add_argument("--output", default="directions.pt")
    e.set_defaults(func=cmd_extract)

    g = sub.add_parser("generate", help="compare baseline vs ablated generation for one prompt")
    _add_common(g)
    g.add_argument("--prompt", required=True)
    g.add_argument("--layer", type=int, default=None, help="direction layer (default: middle)")
    g.add_argument("--alpha", type=float, default=1.0)
    g.add_argument("--min-layer", type=int, default=0)
    g.add_argument("--max-layer", type=int, default=None)
    g.add_argument("--max-new-tokens", type=int, default=128)
    g.set_defaults(func=cmd_generate)

    ev = sub.add_parser("eval", help="benchmark baseline vs ablated ASR/refusal with a judge")
    _add_common(ev)
    ev.add_argument("--benchmark", default="harmbench",
                    help="harmbench|advbench|jailbreakbench|builtin | hf:<ds>:<col>")
    ev.add_argument("--judge", default="keyword",
                    help="keyword | openai:<model> | anthropic:<model> | hf:<model>")
    ev.add_argument("--n", type=int, default=40, help="number of benchmark prompts")
    ev.add_argument("--trials", type=int, default=15)
    ev.add_argument("--subspace", action="store_true")
    ev.add_argument("--n-directions", type=int, default=4)
    ev.add_argument("--max-new-tokens", type=int, default=128)
    ev.set_defaults(func=cmd_eval)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
