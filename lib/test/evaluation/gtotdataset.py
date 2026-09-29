import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class GTOTDataset(BaseDataset):
    """ GTOT dataset for RGB-T tracking.

    Publication:
        LasHeR: A Large-scale High-diversity Benchmark for RGBT Tracking
        Chenglong Li, Wanlin Xue, Yaqing Jia, Zhichen Qu, Bin Luo, Jin Tang, and Dengdi Sun
        https://arxiv.org/abs/2104.13202

    Download dataset from https://github.com/BUGPLEASEOUT/LasHeR
    """
    def __init__(self, split=None):
        super().__init__()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        self.base_path = os.path.join(self.env_settings.gtot_path)

        self.sequence_list = self._get_sequence_list(split)
        self.split = split

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundTruth_v.txt'.format(self.base_path, sequence_name)
        ground_truth_rect = load_text(str(anno_path), delimiter=' ', dtype=np.float64)
        ground_truth_rect[0][2] = ground_truth_rect[0][2] - ground_truth_rect[0][0] +1
        ground_truth_rect[0][3] = ground_truth_rect[0][3] - ground_truth_rect[0][1]+1
        frames_path_i = '{}/{}/i'.format(self.base_path, sequence_name)
        frames_path_v = '{}/{}/v'.format(self.base_path, sequence_name)
        if sequence_name == 'Otcbvs':
            frame_list_i = [frame for frame in os.listdir(frames_path_i) if frame.endswith(".bmp")]
            frame_list_v = [frame for frame in os.listdir(frames_path_v) if frame.endswith(".bmp")]
        else:
            frame_list_i = [frame for frame in os.listdir(frames_path_i) if frame.endswith(".png")]
            frame_list_v = [frame for frame in os.listdir(frames_path_v) if frame.endswith(".png")]
        # frame_list_i.sort(key=lambda f: int(f[1:-5]))
        frame_list_i = sorted(frame_list_i)
        # frame_list_v.sort(key=lambda f: int(f[1:-5]))
        frame_list_v = sorted(frame_list_v)
        frames_list_i = [os.path.join(frames_path_i, frame) for frame in frame_list_i]
        frames_list_v = [os.path.join(frames_path_v, frame) for frame in frame_list_v]
        frames_list = [frames_list_v, frames_list_i]
        return Sequence(sequence_name, frames_list, 'gtot', ground_truth_rect.reshape(-1, 4))
    
    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        with open('{}/list.txt'.format(self.base_path)) as f:
            sequence_list = f.read().splitlines()

        if split == 'ltrval':
            with open('{}/got10k_val_split.txt'.format(self.env_settings.dataspec_path)) as f:
                seq_ids = f.read().splitlines()

            sequence_list = [sequence_list[int(x)] for x in seq_ids]
        return sequence_list
