from torch.utils.data import Dataset, DataLoader
from tests.constants import *
from hvo_sequence.hvo_seq import HVO_Sequence

import os
import pickle
import pandas as pd
import numpy as np

def check_if_passes_filters(df_row, filters):
    meets_filter = []
    for filter_key, filter_values in zip(filters.keys(), filters.values()):
        if filters[filter_key] is not None:
            if df_row.at[filter_key] in filter_values:
                meets_filter.append(True)
            else:
                meets_filter.append(False)
    return all(meets_filter)


class GrooveMidiDataset(Dataset):
    """
    From: https://github.com/behzadhaki/GrooveEvaluator/blob/main/test/feature_extractor_test.py
    """
    def __init__(
            self,
            source_path=PREPROCESSED_DATASET_DIR,
            subset=SUBSET,
            metadata_csv_filename="metadata.csv",
            hvo_pickle_filename="hvo_sequence_data.obj",
            filters=DEFAULT_FILTERS,
            max_len=32
    ):

        data_file = open(os.path.join(source_path, subset, hvo_pickle_filename), 'rb')
        data_set = pickle.load(data_file)
        metadata = pd.read_csv(os.path.join(source_path, subset, metadata_csv_filename))

        self.hvo_sequences = []
        for ix, hvo_seq in enumerate(data_set):
            if len(hvo_seq.time_signatures) == 1:       # ignore if time_signature change happens
                all_zeros = not np.any(hvo_seq.hvo.flatten())
                if not all_zeros:  # Ignore silent patterns
                    if check_if_passes_filters(metadata.loc[ix], filters):
                        # add metadata to hvo_seq scores
                        hvo_seq.drummer = metadata.loc[ix].at["drummer"]
                        hvo_seq.session = metadata.loc[ix].at["session"]
                        hvo_seq.master_id = metadata.loc[ix].at["master_id"]
                        hvo_seq.style_primary = metadata.loc[ix].at["style_primary"]
                        hvo_seq.style_secondary = metadata.loc[ix].at["style_secondary"]
                        hvo_seq.beat_type = metadata.loc[ix].at["beat_type"]
                        # pad with zeros to match max_len
                        pad_count = max(max_len - hvo_seq.hvo.shape[0], 0)
                        hvo_seq.hvo = np.pad(hvo_seq.hvo, ((0, pad_count), (0, 0)), 'constant')
                        hvo_seq.hvo = hvo_seq.hvo [:max_len, :]         # In case, sequence exceeds max_len
                        self.hvo_sequences.append(hvo_seq)

    def __len__(self):
        return len(self.hvo_sequences)

    def __getitem__(self, idx):
        return self.hvo_sequences[idx]