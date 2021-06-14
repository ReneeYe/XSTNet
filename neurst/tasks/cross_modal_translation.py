# Copyright 2020 ByteDance Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Tuple
import numpy as np
import tensorflow as tf
from absl import logging

import neurst.data.dataset_utils as dataset_utils
from neurst.data.data_pipelines import DataPipeline, build_data_pipeline
from neurst.data.data_pipelines.tagged_text_data_pipeline import TaggedTextDataPipeline
from neurst.data.datasets import Dataset
from neurst.layers.metric_layers.token_metric_layers import AudioFramesMetricLayer, SequenceTokenMetricLayer, BatchCountMetricLayer
from neurst.models import build_model
from neurst.metrics import build_metric
from neurst.models.model_utils import deduce_text_length
from neurst.tasks import register_task
from neurst.tasks.task import Task
from neurst.training.training_utils import minimal_multiple
from neurst.utils import compat
from neurst.utils.configurable import deep_merge_dict
from neurst.utils.flags_core import Flag, ModuleFlag
from neurst.tasks.speech2text import create_audio_bucket_boundaries


def get_speech2text_bucket_sizes(args, num_replicas_in_sync):
    audio_bucket_boundaries = create_audio_bucket_boundaries(args["max_audio_src_len"],
                                                             args["batch_bucket_min_audio_src_len"])
    audio_bucket_boundaries[-1] = minimal_multiple(audio_bucket_boundaries[-1], 8)
    batch_size = dataset_utils.adjust_batch_size(
        args["audio_batch_size"],
        args["batch_size_per_gpu"],
        num_replicas_in_sync=num_replicas_in_sync,
        verbose=False)
    batch_size_per_gpu = batch_size // num_replicas_in_sync
    bucket_batch_sizes = [int(batch_size_per_gpu // bound
                              * num_replicas_in_sync) for bound in audio_bucket_boundaries]
    return audio_bucket_boundaries, bucket_batch_sizes


def get_text2text_bucket_sizes(args, num_replicas_in_sync):
    src_text_bucket_boundaries = dataset_utils.create_batch_bucket_boundaries(args["max_text_src_len"])
    bucket_batch_sizes = dataset_utils.adjust_batch_size(
        args["text_batch_size"],
        args["batch_size_per_gpu"],
        bucket_boundaries={"src_text": src_text_bucket_boundaries}
                            if args["batch_by_tokens"] else None,
        boundaries_reduce_to_length_fn=lambda x: max(tf.nest.flatten(x)),
        num_replicas_in_sync=num_replicas_in_sync)
    return src_text_bucket_boundaries, bucket_batch_sizes


def get_speech2text_bucket_size_with_ratio(args, 
                                           audio_bucket_boundaries, 
                                           bucket_batch_sizes):
    frame_transcript_ratio = args.get("experimental_frame_transcript_ratio", None)
    assert frame_transcript_ratio is not None, "define experimental_frame_transcript_ratio, or it will OOM!"
    trans_bucket_boundaries = [
        int(bound / (frame_transcript_ratio + i * (
            args["max_audio_src_len"] / args["max_audio_trg_len"] - frame_transcript_ratio) /
                     len(audio_bucket_boundaries)))
        for i, bound in enumerate(audio_bucket_boundaries)]
    trans_bucket_boundaries = [minimal_multiple(min(i, args["max_audio_trg_len"]), 8) for i in
                               trans_bucket_boundaries]
    num_buckets = len(trans_bucket_boundaries)
    true_trans_bucket_boundaries = []
    num_input_shapes = 0
    for idx, (batc, bound, tbound) in enumerate(zip(bucket_batch_sizes, audio_bucket_boundaries,
                                                    trans_bucket_boundaries)):
        max_trans_len = [tbound,
                         trans_bucket_boundaries[min(idx + 1, len(bucket_batch_sizes) - 1)]]
        num_input_shapes += len(set(max_trans_len))
        true_trans_bucket_boundaries.append(max_trans_len)
    logging.info(f"There are {num_input_shapes} input shapes to be compiled:")
    for idx, (batc, bound, tbound) in enumerate(zip(bucket_batch_sizes, audio_bucket_boundaries,
                                                    true_trans_bucket_boundaries)):
        logging.info(f"   - batch={batc}, maximum-frames={bound}, "
                     f"maximum-transcript-length={set(tbound)}")
    true_trans_bucket_boundaries = tf.constant(true_trans_bucket_boundaries, dtype=tf.int32)
    true_audio_bucket_boundaries = tf.transpose(tf.constant([audio_bucket_boundaries] * 2, dtype=tf.int32))

    return true_audio_bucket_boundaries, true_trans_bucket_boundaries, num_buckets


@register_task(["xm_translation", "xst_translation", "cross_modal_translation", "XModalPretrain"])
class CrossModalTranslation(Task):
    """ Defines the cross-modal(audio & text) pre-train task. """

    def __init__(self, args):
        """ Initializes the task.

        Args:
            args: A dict of model configurations.
        """
        super(CrossModalTranslation, self).__init__(args)
        text_data_pipeline_cls = args.get("text_data_pipeline.class", TaggedTextDataPipeline)
        text_data_pipeline_params = args.get("text_data_pipeline.params", None) or {}
        self._text_data_pipeline = build_data_pipeline(
            text_data_pipeline_cls, **text_data_pipeline_params)
        self._audio_feature_dim = args["audio_feature_dim"]
        self._audio_feature_channels = args["audio_feature_channels"]

    def get_config(self):
        return {
            "text_data_pipeline.class": self._text_data_pipeline.__class__.__name__,
            "text_data_pipeline.params": self._text_data_pipeline.get_config(),
            "audio_feature_dim": self._audio_feature_dim,
            "audio_feature_channels": self._audio_feature_channels
        }

    @staticmethod
    def class_or_method_args():
        this_args = super(CrossModalTranslation, CrossModalTranslation).class_or_method_args()
        this_args.extend([
            ModuleFlag("text_data_pipeline", DataPipeline.REGISTRY_NAME,
                       default=TaggedTextDataPipeline.__name__,
                       help="The text data pipeline."),
            Flag("audio_feature_dim", dtype=Flag.TYPE.INTEGER, default=1,
                 help="The dimension of audio features."),
            Flag("audio_feature_channels", dtype=Flag.TYPE.INTEGER, default=1,
                 help="The number of channels of audio features."),

            Flag("max_audio_src_len", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The maximum source length of training audio frames."),
            Flag("max_text_src_len", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The maximum source length of training text data."),

            Flag("batch_bucket_min_audio_src_len", dtype=Flag.TYPE.INTEGER, default=1000,
                 help="The minimum source length of the training bucket of audio frames."),
            Flag("batch_bucket_min_text_src_len", dtype=Flag.TYPE.INTEGER, default=120,
                 help="The minimum source length of the training bucket of text data."),

            Flag("max_audio_trg_len", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The maximum target length of training audio data."),
            Flag("max_text_trg_len", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The maximum target length of training text data."),

            Flag("truncate_src", dtype=Flag.TYPE.BOOLEAN, default=None,
                 help="Whether to truncate source to max_audio_src_len or max_text_src_len."),
            Flag("truncate_trg", dtype=Flag.TYPE.BOOLEAN, default=None,
                 help="Whether to truncate target to max_audio_trg_len or max_text_trg_len."),

            Flag("experimental_frame_transcript_ratio", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The ratio of the number of frames and its transcript for training batch bucket."),

            Flag("batch_by_frames", dtype=Flag.TYPE.BOOLEAN, default=True,
                 help="Whether to batch the data by audio frames."),
            Flag("audio_batch_size", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The batch size of audio (frames)."),
            Flag("batch_by_tokens", dtype=Flag.TYPE.BOOLEAN, default=True,
                 help="Whether to batch the data by text tokens."),
            Flag("text_batch_size", dtype=Flag.TYPE.INTEGER, default=None,
                 help="The batch size of text (tokens)."),
        ])
        return this_args

    def inputs_signature(self, mode) -> Tuple[dict, dict]:
        """Returns the input dtypes and signatures (from dataset)."""
        dtypes = {"audio": tf.float32, "audio_length": tf.int64,
                  "src_text": tf.int64,
                  "tgt_text": tf.int64, "tgt_lang": tf.int64}

        signatures = {
            "audio": tf.TensorShape([None, None]),
            "audio_length": tf.TensorShape([None, ]),
            "src_text": tf.TensorShape([None, None]),
            "tgt_text": tf.TensorShape([None, None]),
            "tgt_lang": tf.TensorShape([None, None]),
        }

        return dtypes, signatures

    def build_model(self, args, name=None):
        """ Creates the model. """
        model = build_model(args, 
                            {"audio_feature_dim": self._audio_feature_dim,
                             "audio_feature_channels": self._audio_feature_channels},
                            self._text_data_pipeline.meta, 
                            name=name)
        return model

    def example_to_input(self, batch_of_data: dict, mode) -> dict:
        """ Transform the data examples to model acceptable inputs.

        Args:
            batch_of_data: A dict: name -> tf.keras.layers.Input
            mode: The running mode.

        Returns: The input data for model.
        """
        batch = tf.shape(batch_of_data["audio"])[0]

        input_dict = {
            "audio": tf.reshape(batch_of_data["audio"],
                                [batch, -1, self._audio_feature_dim, self._audio_feature_channels]),
            "audio_length": batch_of_data["audio_length"],
            "src_text": batch_of_data["src_text"],
            "src_length": deduce_text_length(batch_of_data["src_text"],
                                             self._text_data_pipeline.meta["pad_id"],
                                             self._text_data_pipeline.meta["padding_mode"]),
            "trg_lang": batch_of_data["tgt_lang"],
        }
        target_bos = batch_of_data["tgt_text"][:, 0]  # dim=1,

        if mode == compat.ModeKeys.INFER:
            input_dict["trg_input"] = target_bos
        else:
            input_dict["trg"] = batch_of_data["tgt_text"]
            input_dict["trg_length"] = deduce_text_length(batch_of_data["tgt_text"],
                                                  self._text_data_pipeline.meta["pad_id"],
                                                  self._text_data_pipeline.meta["padding_mode"])
            input_dict["trg_input"] = tf.concat([tf.expand_dims(target_bos, axis=-1),
                                                 batch_of_data["tgt_text"][:, :-1]], axis=1)

        return input_dict

    def get_data_postprocess_fn(self, mode):
        if mode == compat.ModeKeys.INFER:
            return self._text_data_pipeline.recover
        raise ValueError("No postprocess for TRAIN/EVAL.")

    def get_data_preprocess_fn(self, mode, ds, args=None) -> dict:
        """ Preprocess data sample according to this task.
        Args:
            args: A dict containing dataset arguments. may contains:
                - args["task"] in ["MT","ASR", "ST"]
            mode: A ModeKeys indicating the running mode.
            ds: neurst.data.datasets.XMMultipleDataset

        Returns: A dict, A callable function to collate (process) a data sample.
                map_func["speech2text"][name] = A callable function to process speech2text data
                map_func["text2text"][name] = A callable function to process text2text data
        """

        if args is None:
            args = self._args
        else:
            args = deep_merge_dict(self._args, args, local_overwrite=False)
        
        trunc_audio = args.get("truncate_src", None)
        max_audio_len = args.get("max_audio_src_len", None)
        max_text_src_len = args.get("max_text_src_len", None)
        trunc_text_trg = args.get("truncate_trg", None)
        max_text_trg_len = args.get("max_text_trg_len", None)

        def _process_audio(audio):
            if trunc_audio and max_audio_len:
                audio = audio[:max_audio_len * self._audio_feature_dim * self._audio_feature_channels]
            return audio

        def _process_text(text, tag):
            if isinstance(text, tf.Tensor) and (text.dtype == tf.string):
                text = text.as_string().decode('utf-8')
            if isinstance(text, str):
                text = self._text_data_pipeline.process(text, is_processed=False)
            if mode == compat.ModeKeys.TRAIN and trunc_text_trg and max_text_trg_len:
                if tag == "tgt_text":
                    max_text_len = max_text_trg_len
                elif tag == "src_text":
                    max_text_len = max_text_src_len
                else: # tag in ["src_lang", "tgt_lang"]
                    max_text_len = 10 # only 1 token, set a arbitrary number
                if isinstance(text, tf.Tensor):
                    text = tf.cond(
                        tf.less_equal(tf.size(text), max_text_len), 
                        lambda: text,
                        lambda: tf.concat([text[:(max_text_len - 1)], text[-1:]], axis=0))
                else:
                    if len(text) > max_text_len:
                        text = text[:(max_text_len - 1)] + text[-1:]
            return text

        def _process_lang(lang):
            if not compat.is_tf_tensor(lang) and isinstance(lang, str):
                if not lang.startswith("<"):
                    lang = f"<{lang}>"
                return self._text_data_pipeline.lang2idx(lang)
            return lang

        def _has_lang_tag(text):
            if isinstance(text, tf.Tensor) and (text.dtype == tf.string):
                text = text.as_string()
            if isinstance(text, str):
                return text.startswith("<")
            return True

        def _process_speech2text(data):
            audio = _process_audio(data["audio"])
            lang = data.get("tgt_lang", None)
            ret = {"audio": audio,
                   "audio_length": tf.cast((tf.shape(audio)[0] if isinstance(audio, tf.Tensor)
                                            else audio.shape[0]) // self._audio_feature_dim // self._audio_feature_channels,
                                           dtype=tf.int64),
                   "src_text": data["src_text"]}
            if _has_lang_tag(data["tgt_text"]) or (lang is None):
                ret["tgt_lang"] = [_process_text(data["tgt_text"], "tgt_text")[0]]
                ret["tgt_text"] = _process_text(data["tgt_text"], "tgt_text")
            else:
                ret["tgt_text"] = [_process_lang(lang)] + _process_text(data["tgt_text"], "tgt_text")
            return ret

        def _process_text2text(data):
            ret = {"audio": tf.constant([], dtype=tf.float32),
                   "audio_length": tf.cast(0, dtype=tf.int64)}
            if _has_lang_tag(data["tgt_text"]):
                ret["src_text"] = _process_text(data["src_text"], "src_text")
                ret["tgt_text"] = _process_text(data["tgt_text"], "tgt_text")
                ret["tgt_lang"] = [_process_text(data["tgt_text"], "tgt_text")[0]]
            else:
                ret["src_text"] = [_process_lang(data["src_lang"])] + _process_text(data["src_text"], "src_text")
                ret["tgt_text"] = [_process_lang(data["tgt_lang"])] + _process_text(data["tgt_text"], "tgt_text")
                ret["tgt_lang"] = [_process_lang(data["tgt_lang"])]
            return ret

        preprocess_func_dict = {}
        for ds_type in ds.datasets:
            preprocess_func_dict[ds_type] = {}
            if ds_type == "speech2text":
                for ds_name in ds.datasets[ds_type]:
                    preprocess_func_dict[ds_type][ds_name] = _process_speech2text
            elif ds_type == "text2text":
                for ds_name in ds.datasets[ds_type]:
                    preprocess_func_dict[ds_type][ds_name] = _process_text2text
            else:
                logging.warning("dataset type must be `text2text` or `speech2text` ")

        return preprocess_func_dict

    def create_and_batch_tfds(self, ds: Dataset, mode,
                              args=None, num_replicas_in_sync=1) -> tf.data.Dataset:
        """ Creates a dataset according to the `mode`.

        Args:
            args: A dict containing dataset arguments.
            ds: A neurst.data.datasets.Dataset object. neurst.data.datasets.XMMultipleDataset object
            mode: A ModeKeys indicating the running mode.
            num_replicas_in_sync: The number of GPUs or other workers. We will generate global
                batches, and each global batch is equally divisible by number of replicas.

        Returns:
            A tf.data.Dataset or a INFER_DATA tuple.
        """
        if args is None:
            args = self._args
        else:
            args = deep_merge_dict(self._args, args, local_overwrite=False)

        float_zero = tf.constant(0, dtype=tf.float32)
        int_zero = tf.constant(0, dtype=tf.int64)
        eos = tf.constant(self._text_data_pipeline.meta["eos_id"], dtype=tf.int64)

        padding_values = {"audio": float_zero,
                          "audio_length": int_zero,
                          "src_text": eos,
                          "tgt_text": eos,
                          "tgt_lang": eos}
        
        dataset = ds.build(map_func=self.get_data_preprocess_fn(mode, ds, args),
                           map_output_dtypes=self.inputs_signature(mode)[0],
                           auto_shard=(mode == compat.ModeKeys.TRAIN),
                           shuffle=(mode == compat.ModeKeys.TRAIN))

        if mode != compat.ModeKeys.TRAIN:
            is_s2t = True
            for x in dataset.take(1):
                if tf.size(x["audio"]) == 0:
                    is_s2t = False
            padded_shapes = {"audio_length": [], "tgt_text": [None], "tgt_lang": [None]}
            if is_s2t:
                padded_shapes["audio"] = [None]
                padded_shapes["src_text"] = [tf.constant(1, dtype=tf.int32)]
                return dataset.cache().padded_batch(
                    dataset_utils.adjust_batch_size(args["batch_size"],
                                                    num_replicas_in_sync=num_replicas_in_sync),
                    padded_shapes=padded_shapes,
                    padding_values=padding_values,
                    drop_remainder=False)
            else:
                padded_shapes["audio"] = [tf.constant(8000, dtype=tf.float32)]
                padded_shapes["src_text"] = [None]
                return dataset.cache().padded_batch(
                    dataset_utils.adjust_batch_size(args["batch_size"],
                                                    num_replicas_in_sync=num_replicas_in_sync),
                    padded_shapes=padded_shapes,
                    padding_values=padding_values,
                    drop_remainder=False
                )

        clean_length_dict = {"audio": args["max_audio_src_len"] *
                                      self._audio_feature_dim * self._audio_feature_channels,
                             "audio_length": -1,
                             "src_text": args["max_text_src_len"],
                             "tgt_text": args["max_text_trg_len"],
                             "tgt_lang": -1}
        dataset = dataset.filter(
            lambda data_sample: tf.reduce_all([
                (length == -1) or (length is None) or
                tf.shape(data_sample[k])[0] <= length
                for k, length in clean_length_dict.items()]))

        logging.info("Created training dataset and batchifying...")
        audio_bucket_boundaries, s2t_bucket_batch_sizes = get_speech2text_bucket_sizes(args,
                                                                                       num_replicas_in_sync)
        s2t_audio_bucket_boundaries, s2t_trans_bucket_boundries, s2t_buckets_num = \
            get_speech2text_bucket_size_with_ratio(args, audio_bucket_boundaries, 
                                                   s2t_bucket_batch_sizes)
        s2t_bucket_batch_sizes = tf.constant(s2t_bucket_batch_sizes, dtype=tf.int64)
        audio_bucket_boundaries = tf.constant(audio_bucket_boundaries, dtype=tf.int32)

        text_bucket_boundaries, t2t_bucket_batch_sizes = get_text2text_bucket_sizes(args,
                                                                                    num_replicas_in_sync)
        t2t_bucket_batch_sizes = tf.constant(t2t_bucket_batch_sizes, dtype=tf.int64)
        text_bucket_boundaries = tf.constant(text_bucket_boundaries, dtype=tf.int32)

        t2t_max_trg_len = tf.constant(args["max_text_trg_len"], dtype=tf.int32)
        # make s2t batches
        t2t_bucket_num = tf.constant(len(t2t_bucket_batch_sizes), tf.int64)

        def example_to_bucket_id(examples):
            """Return a tuple bucket_id for the example"""
            is_text2text = tf.equal(tf.cast(examples["audio_length"], tf.int32),
                                    tf.constant(0, dtype=tf.int32))

            def _to_t2t_bucket_id():
                seq_length = tf.size(examples["src_text"])
                conditions_c = tf.less_equal(tf.cast(seq_length, tf.int32),
                                             tf.cast(text_bucket_boundaries, tf.int32))
                return tf.reduce_min(tf.where(conditions_c))

            def _to_s2t_bucket_id():
                conditions_c = tf.logical_and(
                    tf.less_equal(tf.cast(examples["audio_length"], tf.int32), 
                                  s2t_audio_bucket_boundaries),
                    tf.less_equal(tf.size(examples["tgt_text"]),
                                  s2t_trans_bucket_boundries))
                minimum_match = tf.where(conditions_c)[0]

                return (minimum_match[0] * s2t_buckets_num + minimum_match[1]) + t2t_bucket_num

            return tf.cond(is_text2text, _to_t2t_bucket_id, _to_s2t_bucket_id)

        def window_size_fn(bucket_id):
            def t2t_bucket_size():
                return t2t_bucket_batch_sizes[bucket_id]

            def s2t_bucket_size():
                s2t_bucket_id = bucket_id - t2t_bucket_num
                return s2t_bucket_batch_sizes[s2t_bucket_id // s2t_buckets_num]

            return tf.cond(tf.less(bucket_id, t2t_bucket_num),
                           t2t_bucket_size, s2t_bucket_size)

        def batching_fn(bucket_id, grouped_dataset):
            bucket_batch_size = window_size_fn(bucket_id)

            def t2t_shapes():
                ret = {"audio": [tf.constant(5000, dtype=tf.int32)], "audio_length": [],
                       "src_text": [text_bucket_boundaries[bucket_id]],
                       "tgt_text": [t2t_max_trg_len],}
                ret["tgt_lang"] = [1]
                return ret

            def s2t_shapes():
                s2t_bucket_id = bucket_id - t2t_bucket_num
                ret = {"audio": ([audio_bucket_boundaries[s2t_bucket_id // s2t_buckets_num]
                                  * self._audio_feature_dim * self._audio_feature_channels]),
                       "audio_length": [],
                       "src_text": [tf.constant(5, dtype=tf.int32)],
                       "tgt_text": [s2t_trans_bucket_boundries[s2t_bucket_id // s2t_buckets_num][s2t_bucket_id % s2t_buckets_num]],
                       "tgt_lang": [1]}
                return ret

            padded_shapes = tf.cond(tf.less(bucket_id, t2t_bucket_num),
                                    t2t_shapes, s2t_shapes)
            return grouped_dataset.padded_batch(
                bucket_batch_size,
                padded_shapes=padded_shapes,
                padding_values=padding_values,
                drop_remainder=True
            )
        tfds = dataset.apply(tf.data.experimental.group_by_window(
            key_func=example_to_bucket_id, reduce_func=batching_fn,
            window_size=None, window_size_func=window_size_fn))
        return tfds

    def build_metric_layer(self):
        return [AudioFramesMetricLayer("audio"),
                SequenceTokenMetricLayer("trg"), BatchCountMetricLayer("audio")]

    def get_eval_metric(self, args, name="metric", ds=None):
        """ Returns a neurst.metrics.metric.Metric object for evaluation."""
        return build_metric(args[name + ".class"], language=self._text_data_pipeline.meta["language"],
                            **args[name + ".params"])
