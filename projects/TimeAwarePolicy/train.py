"""Command-line entrypoint for TimeAwarePolicy training."""

import json

from core.training.trainer import PolicyTrainer
from projects.TimeAwarePolicy.arguments.training import parse_args


def main(argv=None):
    args = parse_args(argv)
    if args.saving:
        with open(args.json_file_path, "w") as stream:
            json.dump(vars(args), stream, indent=4)

    trainer = PolicyTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
