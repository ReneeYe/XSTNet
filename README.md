# XSTNet: End-to-end Speech Translation via Cross-modal Progressive Training
This is an implementation of paper 
*"End-to-end Speech Translation via Cross-modal Progressive Training"* 
https://arxiv.org/abs/2104.10380 (accepted by Interspeech2021).
The codebase of the implementation is NeurST (https://github.com/bytedance/neurst.git).
NeurST offers several kinds of BLEU scores for *fair comparisons* (refer https://st-benchmark.github.io), 
including case-sensitive/case-insensitive detokenized/tokenized BLEU.

Note: To run the script successfully, Tensorflow 2.3 is recommended. 

**CONTRIBUTION:**
You are also more than welcomed to test our code on your machines, and report feedbacks on results, bugs and performance!


## Overview and Results
The XSTNet model benefits from its three key design aspects:
1. The self supervising pre-trained sub-network (i.e. wav2vec2.0) as the audio encoder, 
2. The **multi-task** training objective to exploit **additional parallel bilingual text**, and 
3. The ***progressive*** training procedure.

### MuST-C En-X dataset
We report **case-sensitive detokenized BLEU** via sacrebleu toolkit.


| Model      | En-De | En-Es | En-Fr | En-It | En-Nl | En-Pt | En-Ro | En-Ru | Avg.  |
| ---------- |:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|XSTNet-base |	25.5 | 29.6  | 36.0  | 25.5  | 30.0  | 31.3  | 25.1  | 16.9  | 29.0  |
|XSTNet	     |  27.8 | 30.8  | 38.0  | 26.4  | 31.2  | 32.4  | 25.7  | 18.5  | 30.3  |

### Augmented LibriSpeech En-Fr dataset (AKA LibriTrans)
We report both **case-sensitive detokenized BLEU** and **case-insensitive tokenized BLEU (as most of the previous works report)**.

| Model      | case-insensitve tokenized BLEU  | case-sensitive detokenized BLEU |
| ---------- | :-----------------------------: | :------------------------------:| 
|XSTNet-base |	             21.0              |               18.8              |
|XSTNet	     |               21.5              |               19.5              |

## Trained Checkpoints
We offer the SentencePiece-token vocabulary, checkpoints of XSTNet (base and expand version), and the TFRecord of the test data.

| **Datasets** |  **Vocab**  | **Model Checkpoints** | **Test Data TFRecord** |
|:--------:|:-------:|:----------:| :--------:|
| En-De    | [Download](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-de/vocab.zip)  |  [Base](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-de/xstnet_base.zip); [Expand](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-de/xstnet_expand.zip)  | [Download](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-de/tst-COMMON.tfrecords-00000-of-00001) |
| En-Es    | [Download](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-es/vocab.zip)  |  [Base](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-es/xstnet_base.zip); [Expand](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-es/xstnet_expand.zip)  | [Download](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-es/tst-COMMON.tfrecords-00000-of-00001) |
| En-Fr    | [Download](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-fr/vocab.zip)  |  [Base](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-fr/xstnet_base.zip); [Expand](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-fr/xstnet_expand.zip)  | [Download](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-fr/tst-COMMON.tfrecords-00000-of-00001) |
| En-It    | [Download](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-it/vocab.zip)  |  [Base](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-it/xstnet_base.zip); [Expand](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-it/xstnet_expand.zip)  | [Download](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-it/tst-COMMON.tfrecords-00000-of-00001) |
| En-Nl    | [Download](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-nl/vocab.zip)  |  [Base](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-nl/xstnet_base.zip); [Expand](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-nl/xstnet_expand.zip)  | [Download](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-nl/tst-COMMON.tfrecords-00000-of-00001) |
| En-Pt    | [Download](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-pt/vocab.zip)  |  [Base](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-pt/xstnet_base.zip); [Expand](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-pt/xstnet_expand.zip)  | [Download](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-pt/tst-COMMON.tfrecords-00000-of-00001) |
| En-Ro    | [Download](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-ro/vocab.zip)  |  [Base](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-ro/xstnet_base.zip); [Expand](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-ro/xstnet_expand.zip)  | [Download](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-ro/tst-COMMON.tfrecords-00000-of-00001) |
| En-Ru    | [Download](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-ru/vocab.zip)  |  [Base](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-ru/xstnet_base.zip); [Expand](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-ru/xstnet_expand.zip)  | [Download](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/mustc_en-ru/tst-COMMON.tfrecords-00000-of-00001) |
|LibriTrans| [Download](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/libritrans/vocab.zip)  |  [Base](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/libritrans/xstnetl_base.zip); [Expand](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/libritrans/xstnet_expand.zip)   | [Download](http://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/interspeech2021/xstnet/libritrans/test.tfrecords-00000-of-00001) |



## Training a Model

### Install NeurST from Source

```bash
git clone https://github.com/ReneeYe/XSTNet.git
cd XSTNet/
pip3 install -e .
```

### Data pre-processing

The data pre-processing is quite similar with [NeurST example on MuST-C](https://github.com/bytedance/neurst/blob/master/examples/speech_to_text/must-c/README.md).

First, download the raw data from https://ict.fbk.eu/must-c/, and save files to ```${DATA_PATH}```.
Then run the following script to extract audio feature. In this work, we use
```bash
bash XSTNet/prepare_data/extract_audio_feature.sh ${DATA_PATH} ${TGT_LANG}
```

We highly recommend you to tokenize the MT text and map the word tokens to IDs aforehand, in order to speed up the training process.
To do this, you need to first prepare the vocabulary, like [SentencePiece](https://github.com/google/sentencepiece) or [BPE](https://github.com/rsennrich/subword-nmt).

To re-implement, we jointly tokenize the bilingual text (En and X) using subword units with a vocabulary size of 10k, learned from [SentencePiece](https://github.com/google/sentencepiece).
We also provide the vocabulary. You may download them, and put at ```./${VOCAB_PATH}```.

```bash
bash XSTNet/prepare_data/preprocess_text.sh ${DATA_PATH} ${VOCAB_PATH} ${TGT_LANG}
``` 
You can also token extra MT data by yourself.

### Download Wav2vec2.0

Since XSTNet uses [Wav2vec2.0](https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md) as the audio encoder. 
To train the model, please download the [pre-trained wav2vec2 model](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt),
and put the model at ```WAV2VEC2_MODEL_PATH```.


### Prepare the configuration files and run training scripts.
The configuration files are: 
- [task config](config/task_config.yml): define *cross_modal_translation* task, including the data pipeline, the batch size, etc..
- [model config](config/model_config.yml): define the structure of XSTNet, including the structure of wav2vec2, Transformer and the convolutional layer in between.
- [training config](config/training_config.yml): define the *trainer*, including loss function, optimizer, learning rate schedule, pretrained model/module, etc..
- [training data config](config/data_config.yml): define the data for training. We highly recommend to make TFRecord first. Remember to turn on "shuffle_dataset".
- [valid config](config/valid_config.yml): define the data for validation and the metric to save the model checkpoints.

We offer the template of the configuration yaml files at [./config/](config). 
Don't forget to define ```\*_TFRECORD_PATH```, ```SPM_SUBTOKENIZER.\*```, ```TRG_LANG```, etc.

```
cat config/task_config.yml config/model_config.yml config/training_config.yml config/data_config.yml > all_configs.yml
bash run.sh --config_paths all_configs.yml --model_dir ${MODEL_CKPT_PATH}
```

### Run valid experiment to select model
```bash
bash run.sh --entry validation --config_paths config/valid_config.yml --model_dir ${MODEL_CKPT_PATH}
```

### Generate and evaluate the model performance.
```bash
bash run --config_paths config/test_config.yml --model_dir ${MODEL_CKPT_PATH}/best_avg
```
add ```--output_file ${RESULT_OUTPUT_PATH}``` if you want to see the generated results.
We provide both *base* and *expand* versions of XSTNet, as well as *TFRecord* for the test data for a fast re-implementation. you may [download](#Trained Checkpoints) them.

## Citation
```
@InProceedings{ye2021end,
  author    = {Rong Ye and Mingxuan Wang and Lei Li},
  booktitle = {Proc. of INTERSPEECH},
  title     = {End-to-end Speech Translation via Cross-modal Progressive Training},
  year      = {2021},
  month     = aug,
}
```
