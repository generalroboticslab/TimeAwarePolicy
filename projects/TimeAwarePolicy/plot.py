"""Unified command-line entrypoint for available result visualizations."""

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workflow",
        choices=("training-curves", "cmdp-solvers", "stagewise-stability"),
    )
    args, remainder = parser.parse_known_args(argv)

    if args.workflow == "training-curves":
        from projects.TimeAwarePolicy.paper.figures import training_curves

        training_curves.main(remainder)
    elif args.workflow == "cmdp-solvers":
        from projects.TimeAwarePolicy.paper.figures import cmdp_solver_comparison

        cmdp_solver_comparison.main(remainder)
    else:
        from projects.TimeAwarePolicy.paper import build_results

        build_results.main(["--stagewise-stability-only", *remainder])


if __name__ == "__main__":
    main()
