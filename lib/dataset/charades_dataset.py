import os
import random
import cv2
import time
import numpy as np
from itertools import chain as chain
import torch
import torch.utils.data
from collections import defaultdict

from . import utils as utils

DATA_MEAN = [0.485, 0.456, 0.406]
DATA_STD  = [0.229, 0.224, 0.225]

class Charades(torch.utils.data.Dataset):
    """
    Charades video loader. Construct the Charades video loader, then sample
    clips from the videos. For training and validation, a single clip is randomly
    sampled from every video with random cropping, scaling, and flipping. For
    testing, multiple clips are uniformaly sampled from every video with uniform
    cropping. For uniform cropping, we take the left, center, and right crop if
    the width is larger than height, or take top, center, and bottom crop if the
    height is larger than the width.
    """

    def __init__(self,
                 mode, 
                 data_dir,
                 num_ensemble_views,
                 num_spatial_crops,
                 data_prefix,
                 num_frames,
                 sample_rate,
                 train_jitter_scales,
                 crop_size,
                 num_classes,
                 random_flip=True,
                 inv_uni_sample=False,
                 num_retries=10):
        """
        Load Charades data (frame paths, labels, etc. ) to a given Dataset object.
        Args:
            dataset (Dataset): a Dataset object to load Charades data to.
            mode (string): 'train', 'val', or 'test'.
        Args:
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Charades ".format(mode)
        self.mode = mode
        self.data_dir = data_dir
        self.num_ensemble_views = num_ensemble_views
        self.num_spatial_crops = num_spatial_crops
        self.data_prefix = data_prefix
        self.num_frames = num_frames
        self.sample_rate = sample_rate
        self.train_jitter_scales = train_jitter_scales
        self.crop_size = crop_size
        self.num_classes = num_classes
        self.random_filp = random_flip
        self.inverse_uniform_sampling = inv_uni_sample

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (num_ensemble_views * num_spatial_crops)
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.data_dir,
            "{}.csv".format("train" if self.mode == "train" else "val"),
        )
        assert os.path.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )
        (self._path_to_videos, self._labels) = utils.load_image_lists(
            path_to_file, self.data_prefix, return_list=True
        )

        if self.mode != "train":
            # Form video-level labels from frame level annotations.
            self._labels = utils.convert_to_video_level_labels(self._labels)

        self._path_to_videos = list(
            chain.from_iterable(
                [[x] * self._num_clips for x in self._path_to_videos]
            )
        )
        self._labels = list(
            chain.from_iterable([[x] * self._num_clips for x in self._labels])
        )
        self._spatial_temporal_idx = list(
            chain.from_iterable(
                [range(self._num_clips) for _ in range(len(self._labels))]
            )
        )

    def get_seq_frames(self, index):
        """
        Given the video index, return the list of indexs of sampled frames.
        Args:
            index (int): the video index.
        Returns:
            seq (list): the indexes of sampled frames from the video.
        """
        temporal_sample_index = (
            -1
            if self.mode in ["train", "val"]
            else self._spatial_temporal_idx[index]
            // self.num_spatial_crops
        )

        video_length = len(self._path_to_videos[index])
        assert video_length == len(self._labels[index])

        clip_length = (self.num_frames - 1) * self.sample_rate + 1
        if temporal_sample_index == -1:
            if clip_length > video_length:
                start = random.randint(video_length - clip_length, 0)
            else:
                start = random.randint(0, video_length - clip_length)
        else:
            gap = float(max(video_length - clip_length, 0)) / (
                self.num_ensemble_views - 1
            )
            start = int(round(gap * temporal_sample_index))

        seq = [
            max(min(start + i * self.sample_rate, video_length - 1), 0)
            for i in range(self.num_frames)
        ]

        return seq

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video frames can be fetched.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): the index of the video.
            time index (zero): The time index is currently not supported.
            {} extra data, currently not supported
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, self._num_yielded = index

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            spatial_sample_index = -1
            min_scale = self.train_jitter_scales[0]
            max_scale = self.train_jitter_scales[1]
            crop_size = self.crop_size

        elif self.mode in ["test"]:
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.num_spatial_crops
            )
            min_scale, max_scale, crop_size = [self.crop_size] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        seq = self.get_seq_frames(index)
        frames = torch.as_tensor(
            utils.retry_load_images(
                [self._path_to_videos[index][frame] for frame in seq],
                self._num_retries,
            )
        )

        label = utils.aggregate_labels(
            [self._labels[index][i] for i in range(seq[0], seq[-1] + 1)]
        )
        label = torch.as_tensor(
            utils.as_binary_vector(label, self.num_classes)
        )

        # Perform color normalization.
        frames = utils.tensor_normalize(
            frames, DATA_MEAN, DATA_STD
        )
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.random_filp,
            inverse_uniform_sampling=self.inverse_uniform_sampling,
        )
        return frames, label

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)