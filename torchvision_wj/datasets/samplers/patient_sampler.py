import os
import re
from torch.utils.data import Sampler
import random
from itertools import repeat

def id_(x):
    return x
def map_(fn, iter):
    # map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]
    return list(map(fn, iter))

class PatientSampler(Sampler):
    def __init__(self, dataset, grp_regex, shuffle=False, quiet=False):
        filenames = dataset.image_names
        # Might be needed in case of escape sequence fuckups
        # self.grp_regex = bytes(grp_regex, "utf-8").decode('unicode_escape')
        assert grp_regex is not None
        self.grp_regex = grp_regex

        # Configure the shuffling function
        self.shuffle: bool = shuffle
        self.shuffle_fn = (lambda x: random.sample(x, len(x))) if self.shuffle else id_

        # print(f"Grouping using {self.grp_regex} regex")
        # assert grp_regex == "(patient\d+_\d+)_\d+"
        # grouping_regex: Pattern = re.compile("grp_regex")
        grouping_regex = re.compile(self.grp_regex)

        stems = [os.path.splitext(filename)[0] for filename in filenames]  # avoid matching the extension
        matches = map_(grouping_regex.match, stems)
        patients = [match.group(1) for match in matches]

        unique_patients = list(set(patients))
        assert len(unique_patients) < len(filenames)
        if not quiet:
            print(f"Found {len(unique_patients)} unique patients out of {len(filenames)} images ; regex: {self.grp_regex}")

        self.idx_map = dict(zip(unique_patients, repeat(None)))
        for i, patient in enumerate(patients):
            if not self.idx_map[patient]:
                self.idx_map[patient] = []

            self.idx_map[patient] += [i]
        # print(self.idx_map)
        assert sum(len(self.idx_map[k]) for k in unique_patients) == len(filenames)

        # print("Patient to slices mapping done")

    def __len__(self):
        return len(self.idx_map.keys())

    def __iter__(self):
        values = list(self.idx_map.values())
        shuffled = self.shuffle_fn(values)
        return iter(shuffled)
