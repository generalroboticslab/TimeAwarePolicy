"""Command-line entrypoint for TimeAwarePolicy evaluation."""

import json
import os

from core.evaluation.evaluator import PolicyEvaluator
from projects.TimeAwarePolicy.arguments.evaluation import get_args
from projects.TimeAwarePolicy.evaluation.stagewise_stability import (
    StagewiseStabilityEvaluationMixin,
)
from real_robot.evaluation import RealRobotEvaluationMixin


class TimeAwarePolicyEvaluator(
    StagewiseStabilityEvaluationMixin,
    RealRobotEvaluationMixin,
    PolicyEvaluator,
):
    """Compose the project protocols with the reusable policy evaluator."""


def main(argv=None):
    args = get_args(argv)
    print(f"###### Evaluation PID is {os.getpid()} ######")
    if args.saving:
        with open(args.json_file_path, "w") as stream:
            json.dump(vars(args), stream, indent=4)

    evaluator = TimeAwarePolicyEvaluator(args)
    evaluator.run()


if __name__ == "__main__":
    main()
