import argparse


def add_shared_args(parser):
    assert isinstance(parser, argparse.ArgumentParser)

    parser.add_argument("--random-seed", type=int, default=None,
                        help="set random seed; default to not set")
    parser.add_argument("--save-dir", type=str, default="./temp",
                        help="set which dictionary to save the experiment result")
    parser.add_argument("--save-name", type=str, default="temp-name",
                        help="set the save name of the experiment result")

    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--lora-cfg-path", type=str, default=None)
    parser.add_argument("--atker-cfg-path", type=str, default=None)

    parser.add_argument("--dataset", type=str, default=None,
                        choices=["advbench", "harmbench",
                                 "advbench-first50", "harmbench-test40", "harmbench-test50",
                                 "alpaca"])

    parser.add_argument("--datacollator", type=str, default=None,
                        choices=["vicuna-chat", "llama2-chat", "llama3-chat", "qwen2-chat"])


def add_train_args(parser):
    assert isinstance(parser, argparse.ArgumentParser)

    parser.add_argument("--trainer-cfg-path", type=str, default=None)

    parser.add_argument("--model-resume-path", type=str, default=None)
    parser.add_argument("--trainer-resume-path", type=str, default=None)

    parser.add_argument("--utilityset", type=str, default=None,
                        choices=["alpaca"])


def add_eval_args(parser):
    assert isinstance(parser, argparse.ArgumentParser)

    parser.add_argument("--model-resume-path", type=str, default=None)
    parser.add_argument("--exp-type", type=str, default=None,
                        choices=["build-alpacaeval", "build-evalset", "judge-evalset"])

    parser.add_argument("--alpacaeval-cfg-path", type=str, default=None)

    parser.add_argument("--evalset-cfg-path", type=str, default=None)
    parser.add_argument("--evalset-path", type=str, default=None)

    parser.add_argument("--judger-cfg-path", type=str, default=None)
