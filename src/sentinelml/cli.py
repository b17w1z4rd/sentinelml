import argparse
import json
from .core import generate, score, train, FEATURES


def main():
    parser = argparse.ArgumentParser(description="Endpoint telemetry ML experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="Generate synthetic grouped telemetry")
    gen.add_argument("--out", default="data/synthetic.csv")
    gen.add_argument("--episodes", type=int, default=600)
    gen.add_argument("--seed", type=int, default=42)
    fit = sub.add_parser("train", help="Train with episode-disjoint splits")
    fit.add_argument("--data", default="data/synthetic.csv")
    fit.add_argument("--model", default="artifacts/model.joblib")
    fit.add_argument("--report", default="artifacts/report.json")
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--min-precision", type=float, default=.8)
    predict = sub.add_parser("score", help="Score a JSON telemetry record")
    predict.add_argument("--model", default="artifacts/model.joblib")
    predict.add_argument("--input", required=True, help="Path to JSON record")
    args = parser.parse_args()
    if args.command == "generate":
        if args.episodes < 30:
            parser.error("--episodes must be >= 30")
        print(generate(args.out, args.episodes, args.seed))
    elif args.command == "train":
        if not 0 < args.min_precision <= 1:
            parser.error("--min-precision must be in (0, 1]")
        print(json.dumps(train(args.data, args.model, args.report, args.seed, args.min_precision), indent=2))
    else:
        with open(args.input) as f:
            record = json.load(f)
        print(json.dumps(score(args.model, record), indent=2))

if __name__ == "__main__":
    main()
