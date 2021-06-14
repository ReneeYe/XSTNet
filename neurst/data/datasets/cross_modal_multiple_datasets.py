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

import tensorflow as tf
from absl import logging
import random

from neurst.data.dataset_utils import glob_tfrecords, load_tfrecords
from neurst.data.datasets import Dataset, register_dataset
from neurst.utils.compat import DataStatus
from neurst.utils.flags_core import Flag
from neurst.utils.misc import to_numpy_or_python_type


def text_is_processed(example, key):
    """
    Args:
        example: tf.train.Example()
        key: str
    Return: the status in DataStatus
    """
    if len(example.features.feature[key].bytes_list.value) > 0:
        return DataStatus.RAW
    elif len(example.features.feature[key].int64_list.value) > 0:
        return DataStatus.PROJECTED
    else:
        return DataStatus.RAW


def text_has_lang_tag(example, key):
    text = example.features.feature[key].bytes_list.value[0].decode("utf-8")
    if text.startswith("<"):
        return True
    return False

@register_dataset(["xm_multiple_datasets", "XMMultipleDatasets", "cross_modal_multiple_dataset"])
class CrossModalMultipleDataset(Dataset):

    def __init__(self, args):
        """ Initializes the multiple dataset.

        Args:
            args: containing `multiple_dataset`, which is like
                {
                    "data0": {"data_path": "", "src_key": "", "tgt_key":""...},
                    "data1": {"data_path": "", "src_key": "", "tgt_key":""...},
                    ......
                }
        """
        super(CrossModalMultipleDataset, self).__init__()
        self._shuffle_dataset = args["shuffle_dataset"]
        self._datasets = {"text2text": {}, "speech2text": {}}
        self._targets = None
        for name, dargs in args["xm_multiple_datasets"].items():
            if "MT" in name: # it is a text2text dataset
                self._datasets["text2text"][name] = {}
                target_tag = "tgt"
                if "data_path" in dargs:
                    for x in tf.data.TFRecordDataset(glob_tfrecords((dargs["data_path"]))).take(1):
                        example = tf.train.Example()
                        example.ParseFromString(x.numpy())
                        if "tgt_lang" in example.features.feature:
                            target_tag = "tgt"
                        elif "trg_lang" in example.features.feature:
                            target_tag = "trg"
                        dargs["tgt_status"] = text_is_processed(example, dargs["tgt_key"])
                        dargs["src_status"] = text_is_processed(example, dargs["src_key"])
                        if dargs["tgt_status"] == DataStatus.RAW:
                            dargs["tgt_has_langtag"] = text_has_lang_tag(example, dargs["tgt_key"])
                    if ("src_file" in dargs) and ("tgt_file" in dargs):
                        logging.info("Check if it is a validation or generation")
                        self._datasets["text2text"][name]["src_file"] = dargs["src_file"]
                        self._datasets["text2text"][name]["tgt_file"] = dargs["tgt_file"]
                else:
                    logging.info(f"`data_path` not found, make to TFRecord first, skip {name}")
                    continue
                self._datasets["text2text"][name]["args"] = dargs
                self._datasets["text2text"][name]["data_field"] = \
                    {"audio": tf.io.VarLenFeature(tf.float32),
                     dargs["src_key"]: tf.io.VarLenFeature(tf.int64) if dargs["src_status"] == DataStatus.PROJECTED
                     else tf.io.VarLenFeature(tf.string),
                     dargs["tgt_key"]: tf.io.VarLenFeature(tf.int64) if dargs["tgt_status"] == DataStatus.PROJECTED
                     else tf.io.VarLenFeature(tf.string)}
                self._datasets["text2text"][name]["feature_name_mapping"] = {dargs["src_key"]: "src_text",
                                                                             dargs["tgt_key"]: "tgt_text"}
                if not dargs.get("tgt_has_langtag", True):
                    self._datasets["text2text"][name]["data_field"]["src_lang"] = tf.io.VarLenFeature(tf.string)
                    self._datasets["text2text"][name]["data_field"][f"{target_tag}_lang"] = tf.io.VarLenFeature(tf.string)
                    self._datasets["text2text"][name]["feature_name_mapping"]["src_lang"] = "src_lang"
                    self._datasets["text2text"][name]["feature_name_mapping"][f"{target_tag}_lang"] = "tgt_lang"

            elif ("ASR" in name) or ("ST" in name):  # it is a speech2text dataset
                self._datasets["speech2text"][name] = {}
                target_tag = "tgt"
                if "data_path" in dargs:
                    for x in tf.data.TFRecordDataset(glob_tfrecords((dargs["data_path"]))).take(1):
                        example = tf.train.Example()
                        example.ParseFromString(x.numpy())
                        if "tgt_lang" in example.features.feature:
                            target_tag = "tgt"
                        elif "trg_lang" in example.features.feature:
                            target_tag = "trg"
                        dargs["src_status"] = DataStatus.PROJECTED
                        dargs["tgt_status"] = text_is_processed(example, dargs["tgt_key"])
                        if dargs["tgt_status"] == DataStatus.RAW:
                            dargs["tgt_has_langtag"] = text_has_lang_tag(example, dargs["tgt_key"])
                else:
                    logging.info(f"`data_path` not found, make to TFRecord first, skip {name}")
                    continue
                self._datasets["speech2text"][name]["args"] = dargs
                self._datasets["speech2text"][name]["data_field"] = \
                    {dargs["src_key"]: tf.io.VarLenFeature(tf.float32),
                     "src_text": tf.io.VarLenFeature(tf.int64),
                     dargs["tgt_key"]: tf.io.VarLenFeature(tf.int64) if dargs["tgt_status"] == DataStatus.PROJECTED
                     else tf.io.VarLenFeature(tf.string)}
                self._datasets["speech2text"][name]["feature_name_mapping"] = {dargs["src_key"]: "audio",
                                                                               dargs["tgt_key"]: "tgt_text"}
                if not dargs.get("tgt_has_langtag", True):
                    if dargs["tgt_key"] == "translation":
                        self._datasets["speech2text"][name]["feature_name_mapping"][f"{target_tag}_lang"] = "tgt_lang"
                        self._datasets["speech2text"][name]["data_field"][f"{target_tag}_lang"] = tf.io.VarLenFeature(tf.string)
                    elif dargs["tgt_key"] == "transcript":
                        self._datasets["speech2text"][name]["feature_name_mapping"]["src_lang"] = "tgt_lang"
                        self._datasets["speech2text"][name]["data_field"]["src_lang"] = tf.io.VarLenFeature(tf.string)
            else:
                logging.info(f"Dataset `name`({name}) must have keywords: ASR/ST/MT. Omitted")
                continue

    @staticmethod
    def class_or_method_args():
        return [
            Flag("shuffle_dataset", dtype=Flag.TYPE.BOOLEAN,
                 help="Whether to shuffle the TF records files. "
                      "Note that some parts may be lost under MultiWorkerMirroredStrategy if set True."),

            Flag("xm_multiple_datasets", dtype=Flag.TYPE.STRING,
                 help="A dict of dataset class and parameters, "
                      "where the key is the dataset name and "
                      "the value is a dict of arguments for one dataset."),
        ]

    @property
    def status(self):
        # return a dict: name -> projected/raw/processed
        status_dict = {}
        for ds_type in self.datasets:
            for ds_name in self.datasets[ds_type]:
                status_dict[ds_name] = self.datasets[ds_type][ds_name].get("status", DataStatus.RAW)
        logging.info("status:", status_dict)
        return status_dict

    @property
    def fields(self):
        return None
    
    @property
    def dataset_type(self):
        return ["text2text", "speech2text"]
    
    @property
    def datasets(self):
        return self._datasets

    def build(self, auto_shard=False, map_func=None, map_output_dtypes=None,
              shuffle=True) -> tf.data.Dataset:
        """
        Args:
            map_func is a dict, {"speech2text": {"name0": map_func0}..., "text2text": {"name1": map_func1}...}
        Return: a dict:
            {"speech2text": {"name0": tf.data.Datasets} ..., "text2text": {"name1": tf.data.Datasets}... }
        """

        def _gen(d_type, name):
            d_type = d_type.decode("utf-8")
            name = name.decode("utf-8")
            iterator = load_tfrecords(self.datasets[d_type][name]["args"]["data_path"],
                                      shuffle=self._shuffle_dataset and shuffle,
                                      deterministic=(not shuffle),
                                      auto_shard=auto_shard,
                                      name_to_features=self.datasets[d_type][name]["data_field"],
                                      feature_name_mapping=self.datasets[d_type][name]["feature_name_mapping"])
            for data in iterator:
                data = to_numpy_or_python_type(data, bytes_as_str=True)
                if map_func[d_type][name] is not None:
                    data = map_func[d_type][name](data)
                yield data

        dataset_dict = {}
        for d_type in self.dataset_type:
            if len(self.datasets[d_type]) > 0:
                if d_type not in dataset_dict:
                    dataset_dict[d_type] = {}
                for name, item in self.datasets[d_type].items():
                    tfds = None
                    ds_args = item.get("args", None)
                    if ds_args is not None:
                        try:
                            logging.info(f"Call load_tfrecords for {d_type} dataset `{name}`")
                            tfds = load_tfrecords(
                                self.datasets[d_type][name]["args"]["data_path"],
                                shuffle=self._shuffle_dataset and shuffle,
                                deterministic=(not shuffle),
                                map_func=lambda x: map_func[d_type][name](x),
                                auto_shard=auto_shard,
                                name_to_features=self.datasets[d_type][name]["data_field"],
                                feature_name_mapping=self.datasets[d_type][name]["feature_name_mapping"])
                        except AttributeError:
                            logging.info(f"Call Dataset.from_generator for {d_type} dataset `{name}`")
                            tfds = tf.data.Dataset.from_generator(_gen,
                                                                  args=(d_type, name,),
                                                                  output_types=map_output_dtypes)
                        except:
                            logging.info(f"load_tfrecords for dataset `{name}` failed, skip it")
                            continue

                    if tfds:
                        dataset_dict[d_type][name] = tfds

        dataset = None
        if self._shuffle_dataset:
            has_value = {}
            total_sample_num = {}
            dataset_iter = {}
            for dstype in dataset_dict:
                has_value[dstype] = {}
                total_sample_num[dstype] = {}
                dataset_iter[dstype] = {}
                for dsname in dataset_dict[dstype]:
                    dataset_iter[dstype][dsname] = iter(dataset_dict[dstype][dsname])
                    # dataset_iter[dstype][dsname] = dataset_dict[dstype][dsname].as_numpy_iterator()
                    has_value[dstype][dsname] = True

            def _final_gen():
                # while any([any(has_value[dtype].values()) for dtype in has_value.keys()]):
                while True:
                    ds_type = random.choice(list(has_value.keys()))
                    ds_name = random.choice(list(has_value[ds_type]))
                    try:
                        sample = next(dataset_iter[ds_type][ds_name])
                        yield sample
                    except StopIteration:
                        has_value[ds_type][ds_name] = False
                        try:
                            data = load_tfrecords(
                                self.datasets[ds_type][ds_name]["args"]["data_path"],
                                shuffle=self._shuffle_dataset and shuffle,
                                deterministic=(not shuffle),
                                map_func=lambda x: map_func[ds_type][ds_name](x),
                                auto_shard=auto_shard,
                                name_to_features=self.datasets[ds_type][ds_name]["data_field"],
                                feature_name_mapping=self.datasets[ds_type][ds_name]["feature_name_mapping"])
                        except AttributeError:
                            data = tf.data.Dataset.from_generator(_gen,
                                                                  args=(ds_type, ds_name,),
                                                                  output_types=map_output_dtypes)
                        dataset_iter[ds_type][ds_name] = data.as_numpy_iterator()

            dataset = tf.data.Dataset.from_generator(_final_gen,
                                                     output_types=map_output_dtypes)
        else:
            for ds_type in dataset_dict:
                for ds_name in dataset_dict[ds_type]:
                    if dataset is None:
                        dataset = dataset_dict[ds_type][ds_name]
                    else:
                        dataset = dataset.concatenate(dataset_dict[ds_type][ds_name])
        return dataset

    def build_iterator(self, map_func=None, shard_id=0, total_shards=1,
                       *args, **kwargs):

        def gen():
            ds = load_tfrecords(self._valid_ds_args["data_path"], shuffle=False, auto_shard=False,
                                name_to_features=self._valid_ds_data_field,
                                sharding_index=shard_id, num_shards=total_shards,
                                feature_name_mapping=self._valid_ds_feature_name_mapping)
            for x in ds:
                data = to_numpy_or_python_type(x, bytes_as_str=True)
                if map_func is not None:
                    data = map_func(data)
                yield data

        return gen

    @property
    def targets(self):
        """ Returns a list of targets. """
        if self._targets is None:
            data_type = ""
            if len(self._datasets["speech2text"]) == 1:
                data_type = "speech2text"
            elif len(self._datasets["text2text"]) == 1:
                data_type = "text2text"
            else:
                assert len(self._datasets["speech2text"]) == 1 or len(self._datasets["text2text"]) == 1
            self._valid_ds_name = list(self._datasets[data_type].keys())[0]
            self._valid_ds_args = list(self._datasets[data_type].values())[0]["args"]
            self._valid_ds_data_field = list(self._datasets[data_type].values())[0]["data_field"]
            self._valid_ds_feature_name_mapping = list(self._datasets[data_type].values())[0]["feature_name_mapping"]

            print(self._datasets)

            if data_type == "speech2text":
                if self._valid_ds_args.get("tgt_language", None) is None:
                    gen = self.build_iterator(map_func=lambda x: " ".join(x["tgt_text"].split()[1:]), )
                else:
                    gen = self.build_iterator(map_func=lambda x: x["tgt_text"])
                self._targets = [x for x in gen()]
            else:
                with tf.io.gfile.GFile(self._valid_ds_args["tgt_file"]) as fp:
                    self._targets = [" ".join(line.strip().split()[1:]) for line in fp]

        return self._targets

