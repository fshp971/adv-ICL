import os, numpy as np, json
from datasets import load_dataset

from adv_trainers.data import DatasetWrapper


def advbench(root="../data"):
    path = os.path.join(root, "advbench/harmful_behaviors.csv")
    # use all samples as train samples for now
    ds = load_dataset("csv", data_files=path)["train"]
    return DatasetWrapper(ds, prompt_name="goal", target_name="target")


def advbench_first50(root="../data"):
    path = os.path.join(root, "advbench/harmful_behaviors.csv")
    ds = load_dataset("csv", data_files=path)["train"]
    ds = [ds[i] for i in range(50)]
    return DatasetWrapper(ds, prompt_name="goal", target_name="target")


def alpaca():
    ds = load_dataset("tatsu-lab/alpaca")["train"]
    new_ds = []

    for sample in ds:
        new_sample = {
            "instruction": sample["instruction"],
            "output": sample["output"],
        }
        if sample["input"] != '':
            new_sample["instruction"] += '\n\n' + sample["input"]
        new_ds.append(new_sample)

    return DatasetWrapper(new_ds, prompt_name="instruction", target_name="output")


class HarmBench:
    def __init__(self, root: str = "../data", split: str = "train"):
        self.root = root
        self.split = split
        self.is_train = (split == "train")

        if self.split == "train":
            prompt_path = "harmbench/behavior_datasets/extra_behavior_datasets/adv_training_behaviors.csv"
            self.prompt_path = os.path.join(root, prompt_path)
            target_path = "harmbench/optimizer_targets/extra_targets/adv_training_targets.json"
            self.target_path = os.path.join(root, target_path)

        elif self.split == "test":
            prompt_path = "harmbench/behavior_datasets/harmbench_behaviors_text_test.csv"
            self.prompt_path = os.path.join(root, prompt_path)
            target_path = "harmbench/optimizer_targets/harmbench_targets_text.json"
            self.target_path = os.path.join(root, target_path)

        else:
            raise ValueError

        self.prompt_ds = load_dataset("csv", data_files=self.prompt_path)["train"]
        with open(self.target_path, "rb") as f:
            self.target_ds = json.load(f)

    def __getitem__(self, idx):
        sample = self.prompt_ds[idx]

        if self.is_train:
            behavior_list = self.target_ds[sample["BehaviorID"]]
            return {
                "prompt": sample["Behavior"],
                "target": behavior_list[ np.random.randint(len(behavior_list)) ],
            }

        return {
            "prompt": sample["Behavior"],
            "target": self.target_ds[sample["BehaviorID"]],
        }

    def __len__(self):
        return len(self.prompt_ds)

def harmbench():
    ds = HarmBench()
    return DatasetWrapper(ds, prompt_name="prompt", target_name="target")


class HarmBenchTest40(HarmBench):
    def __init__(self, root: str = "../data"):
        super().__init__(root=root, split="test")

    def __len__(self):
        return 40

def harmbench_test_40():
    ds = HarmBenchTest40()
    return DatasetWrapper(ds, prompt_name="prompt", target_name="target")


class HarmBenchTest50(HarmBench):
    def __init__(self, root: str = "../data"):
        super().__init__(root=root, split="test")

    def __len__(self):
        return 50

def harmbench_test_50():
    ds = HarmBenchTest50()
    return DatasetWrapper(ds, prompt_name="prompt", target_name="target")


__dataset_zoo__ = {
    "advbench"         : advbench,
    "advbench-first50" : advbench_first50,
    "alpaca"           : alpaca,
    "harmbench"        : harmbench,
    "harmbench-test40" : harmbench_test_40,
    "harmbench-test50" : harmbench_test_50,
}

def get_dataset(name):
    return __dataset_zoo__[name]()
