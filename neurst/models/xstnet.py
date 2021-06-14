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

from neurst.models import build_model, register_model
from neurst.models.model import BaseModel
from neurst.models.wav2vec2 import Wav2Vec2

from neurst.layers.modalities.audio_modalities import Conv1dSubsampling
from neurst.layers.modalities.text_modalities import WordEmbeddingSharedWeights
from neurst.layers.decoders import build_decoder
from neurst.layers.encoders import build_encoder
from neurst.layers.common_layers import PositionEmbeddingWrapper

from neurst.models.model_utils import input_length_to_padding
from neurst.utils import compat
from neurst.utils.flags_core import Flag
from neurst.utils.hparams_sets import get_hyper_parameters


@register_model(["XSTNet", "xstnet", "cross_speech_text_network", "cross_modal_transformer", "CrossModalTransformer"])
class XSTNet(BaseModel):
    """ Defines the CrossModalTransformer model. i.e.
            1) Wav2vec2.0 + CNN + Transformer for speech2text data
            2) Transformer for text2text data
    """
    def __init__(self,
                 args,
                 audio_meta, 
                 text_meta,
                 wav2vec2_model,
                 audio_modality,
                 text_modality,
                 transformer_encoder,
                 transformer_decoder,
                 name=None):
        super(XSTNet, self).__init__(
            args, name=(name or "cross_modal_transformer"))
        self._wav2vec2_model = wav2vec2_model
        self._audio_meta = audio_meta
        self._text_meta = text_meta
        self._audio_modality = audio_modality
        self._text_modality = text_modality
        self._transformer_encoder = transformer_encoder
        self._transformer_decoder = transformer_decoder
        self._output_logits_layer = tf.keras.layers.Dense(
            text_meta["vocab_size"], activation=None, 
            use_bias=True, name="softmax_linear")
        # zero audio init
        self.zero_audio = tf.zeros([1, 5000], tf.float16)
        self.zero_audio_length = tf.constant([5000], tf.int64)

    @staticmethod
    def class_or_method_args():
        return [
            Flag("wav2vec2_params", dtype=Flag.TYPE.STRING, default="wav2vec2_base",
                 help="A string indicating the hyper parameters set of the wav2vec feature extractor, "
                      "or a json/yaml-like string defining the full parameters of Wav2Vec2 model."),
            Flag("transformer_params", dtype=Flag.TYPE.STRING, default="transformer_base",
                 help="A string indicating the hyper parameters set of the transformer"),
            Flag("audio_conv.kernel_size", dtype=list, default=[3, 3]),
            Flag("audio_conv.strides", dtype=list, default=[2, 2]),
            Flag("audio_conv.channels", dtype=Flag.TYPE.INTEGER, default=256),
            Flag("audio_conv.dropout", dtype=Flag.TYPE.FLOAT, default=0.0),
            Flag("audio_conv.layer_norm", dtype=Flag.TYPE.BOOLEAN, default=True),
            Flag("modality.audio.timing", dtype=Flag.TYPE.STRING, default="sinusoids"),
            Flag("modality.text.timing", dtype=Flag.TYPE.STRING, default="sinusoids"),
        ]

    @classmethod
    def new(cls, args: dict, audio_meta, text_meta, name=None):
        """ Builds cross-modal transformer.

        Args:
            args: A dict containing all model parameters.
            audio_meta: A dict containing audio-side vocabulary meta data, e.g. audio feature length.
            text_meta: A dict containing text-side vocabulary meta data, e.g. eos_id, vocab_size.
            name: The name of the model.

        Returns:
            An encoder decoder model.
        """
        with tf.name_scope("cross_modal_transformer"):
            # PART1: Wav2vec2.0
            _wav2vec2_args = args["wav2vec2_params"]
            if isinstance(_wav2vec2_args, str):
                _wav2vec2_args = get_hyper_parameters(_wav2vec2_args)["model.params"]
            assert isinstance(_wav2vec2_args, dict)
            args["wav2vec2_params"] = _wav2vec2_args
            wav2vec2_model = build_model(Wav2Vec2, name="wav2vec2", **_wav2vec2_args)

            # PART2: Audio modality: CNN layers
            audio_meta["audio_feature_dim"] = wav2vec2_model.args["encoder_embed_dim"]
            audio_meta["audio_feature_channels"] = 1
            audio_modality = cls.build_audio_modality(args)
            # PART3: text_modality: word_embeddings
            _transformer_args = args["transformer_params"]

            def _define_dmodel(transformer_args):
                dmodel = transformer_args["modality.dim"]
                if dmodel != args["audio_conv.channels"]:
                    logging.info("Transformer`s d_model not equal CNN`s channel size, "
                                 "rewrite the config of Transformer")
                    dmodel = args["audio_conv.channels"]
                    transformer_args["modality.dim"] = dmodel
                    transformer_args["encoder.hidden_size"] = dmodel
                    transformer_args["decoder.hidden_size"] = dmodel
                return transformer_args, dmodel

            if isinstance(_transformer_args, str):
                _transformer_args = get_hyper_parameters(_transformer_args)["model.params"]
            assert isinstance(_transformer_args, dict)
            _transformer_args, args["modality.text.dim"] = _define_dmodel(_transformer_args)
            args["transformer_params"] = _transformer_args

            text_modality = cls.build_text_modality(args, text_meta)

            # PART4: Transformer
            transformer_encoder_params = {}
            transformer_decoder_params = {}
            for f in _transformer_args:
                if f.startswith("encoder."):
                    transformer_decoder_params[f[8:]] = _transformer_args[f]
                elif f.startswith("decoder."):
                    transformer_encoder_params[f[8:]] = _transformer_args[f]
            transformer_encoder = build_encoder({
                "encoder.class": "TransformerEncoder",
                "encoder.params": transformer_encoder_params}, name="TransformerEncoder")
            transformer_decoder = build_decoder({
                "decoder.class": "TransformerDecoder",
                "decoder.params": transformer_decoder_params}, name="TransformerDecoder")

            model = cls(args, audio_meta, text_meta, wav2vec2_model, audio_modality, text_modality,
                        transformer_encoder, transformer_decoder)

            s2t_input = {"audio": tf.random.normal([2, 10000, 1, 1], dtype=tf.float32),
                         "audio_length": tf.convert_to_tensor([10000, 10000], tf.int64),
                         "src_text": tf.convert_to_tensor([[5],
                                                           [5]], tf.int64),
                         "src_length": tf.convert_to_tensor([1, 1], tf.int64),
                         "trg_input": tf.convert_to_tensor([[1, 2, 3],
                                                            [3, 4, 1]], tf.int64),
                         "trg_lang": tf.convert_to_tensor([[1], [1]], tf.int64)}

            t2t_input = {"audio": tf.reshape(tf.zeros([2, 5000], tf.float16), [2, -1, 1, 1]),
                         "audio_length": tf.convert_to_tensor([0, 0], tf.int64),
                         "src_text": tf.convert_to_tensor([[1, 3, 2, 4],
                                                           [1, 9, 0, 0]], tf.int64),
                         "src_length": tf.convert_to_tensor([4, 2], tf.int64),
                         "trg_input": tf.convert_to_tensor([[1, 2, 3],
                                                            [3, 4, 1]], tf.int64),
                         "trg_lang": tf.convert_to_tensor([[3], [3]], tf.int64)}

            _ = model(s2t_input)
            _ = model(t2t_input)
        return model

    @classmethod
    def build_audio_modality(cls, args):
        with tf.name_scope("audio_modality"):
            audio_modality = Conv1dSubsampling(
                channels=args["audio_conv.channels"],
                strides=tuple(args["audio_conv.strides"]),
                kernel_sizes=tuple(args["audio_conv.kernel_size"]),
                layer_dropout=args["audio_conv.dropout"],
                layer_norm=args["audio_conv.layer_norm"],
                name="audio_conv"
            )

        if args["modality.audio.timing"]:
            src_timing = {"timing": args["modality.audio.timing"]}
            audio_modality = PositionEmbeddingWrapper(
                embedding_layer=audio_modality, name="input_audio_modality_posenc_wrapper", **src_timing)
        return audio_modality
        
    @classmethod
    def build_text_modality(cls, args, text_meta):
        """to build modality for text, source and target embedding modalities are shared.
        """
        text_modality = WordEmbeddingSharedWeights(
            embedding_dim=args["modality.text.dim"], vocab_size=text_meta["vocab_size"],
            share_softmax_weights=False,
            name="text_symbol_modality")
        if args["modality.text.timing"]:
            timing = {"timing": args["modality.text.timing"]}
            text_modality = PositionEmbeddingWrapper(
                embedding_layer=text_modality, name="text_symbol_modality_posenc_wapper", **timing)
        return text_modality

    def get_symbols_to_logits_fn(self, inputs, is_training, is_inference,
                                 decode_padded_length=None,
                                 *args, **kwargs):
        """Prepare for decoding
        Args:
            inputs: a dict of all model inputs, including: audio, audio_length, src_text, trg_input, trg_padding
        """
        batch_size = tf.shape(inputs["trg_input"])[0]
        audio = inputs.pop("audio")
        audio_length = inputs.pop("audio_length")
        src_text = inputs.pop("src_text")
        src_length = inputs.pop("src_length")

        # process text2text
        text_embedded_inputs = tf.cast(self._text_modality(src_text),
                                       dtype=tf.dtypes.as_dtype(compat.CUSTOM_GLOBAL_FLOATX))
        
        text_src_padding = input_length_to_padding(src_length, tf.shape(text_embedded_inputs)[1])
        text_encoder_output = self._transformer_encoder(text_embedded_inputs, text_src_padding,
                                                        is_training=is_training)
        text_decoder_internal_cache = self._transformer_decoder.create_decoding_internal_cache(
            encoder_outputs=text_encoder_output,
            encoder_inputs_padding=text_src_padding,
            is_inference=is_inference)

        # process speech2text
        audio_src = tf.squeeze(tf.squeeze(audio, -1), -1) # [batch, frames]
        wav2vec2_out = self._wav2vec2_model({
            "src": audio_src,
            "src_padding": (1. - tf.sequence_mask(
                    lengths=tf.cast(audio_length, tf.int32),
                    maxlen=tf.cast(tf.shape(audio_src)[1], tf.int32),
                    dtype=tf.dtypes.as_dtype(compat.CUSTOM_GLOBAL_FLOATX)))
        }, is_training=is_training)
        audio_src_after_w2v = wav2vec2_out["contextualized_representation"]
        # batch x short_len_1 x w2v_hidden
        audio_src_length_after_w2v = (tf.shape(wav2vec2_out["contextualized_representation_padding"])[1]
                                      - tf.cast(tf.reduce_sum(wav2vec2_out["contextualized_representation_padding"],
                                                axis=-1), tf.int32))

        def _length_after_conv(_l):
            ans = _l
            for i in range(len(self.args["audio_conv.kernel_size"])):
                ans = tf.math.floordiv(ans + (self.args["audio_conv.kernel_size"][i] // 2) * 2
                                           - self.args["audio_conv.kernel_size"][i],
                                       self.args["audio_conv.strides"][i]) + 1
            return ans

        audio_embedded_inputs = tf.cast(self._audio_modality(audio_src_after_w2v, is_training=is_training),
                                        dtype=tf.dtypes.as_dtype(compat.CUSTOM_GLOBAL_FLOATX))
        # batch x short_len_2 x d_model_text(cnn_channel)

        # add embedding of <audio> tag
        # tf.repeat(self.zero_audio, batch_size, axis=0)
        # audio_tags = inputs.pop("trg_lang")
        audio_tags = tf.repeat(
            tf.reshape(tf.convert_to_tensor(self._text_meta["tag_dict"]["<audio>"], tf.int64),
                       [1, 1]), batch_size, axis=0) # b x 1
        audio_tags_emb = tf.cast(self._text_modality(audio_tags),
                                 dtype=tf.dtypes.as_dtype(compat.CUSTOM_GLOBAL_FLOATX)) # b x 1 x emb
        tagged_audio_embedded_inputs = tf.concat([audio_tags_emb, audio_embedded_inputs], axis=1)

        audio_src_padding = 1. - tf.sequence_mask(
            lengths=1 + tf.cast(_length_after_conv(audio_src_length_after_w2v), tf.int32),
            maxlen=1 + tf.cast(_length_after_conv(tf.shape(audio_src_after_w2v)[1]), tf.int32),
            dtype=tf.dtypes.as_dtype(compat.CUSTOM_GLOBAL_FLOATX)) # remember +1 on length to compute the padding
        audio_encoder_outputs = self._transformer_encoder(tagged_audio_embedded_inputs, audio_src_padding,
                                                          is_training=is_training) # b x frame x hid

        audio_decoder_internal_cache = self._transformer_decoder.create_decoding_internal_cache(
            encoder_outputs=audio_encoder_outputs,
            encoder_inputs_padding=audio_src_padding,
            is_inference=is_inference,
            decode_padded_length=decode_padded_length)

        def symbols_to_logits_fn(symbols, cache, time=None):
            inp = self._text_modality(symbols, time=time)
            if decode_padded_length is None:
                decoder_output = self._transformer_decoder(inp, cache, is_training=is_training,
                                                           decode_loop_step=None)
            else:
                decoder_output = self._transformer_decoder(inp, cache, is_training=is_training,
                                                           decode_loop_step=time)
            logits = self._output_logits_layer(decoder_output)
            return logits

        generation_initializer = {
            "is_text2text": tf.equal(tf.reduce_sum(audio),
                                     tf.constant(0.0, tf.dtypes.as_dtype(compat.CUSTOM_GLOBAL_FLOATX))),
            "decoder_input": inputs["trg_input"],
            "text_decoder_internal_cache": text_decoder_internal_cache,
            "audio_decoder_internal_cache": audio_decoder_internal_cache,
            "text_encoder_inputs_maxlen": tf.shape(text_encoder_output)[1],
            "audio_encoder_inputs_maxlen": tf.shape(audio_encoder_outputs)[1],
            "eos_id": self._text_meta["eos_id"],
            "unk_id": self._text_meta["unk_id"]
        }
        return symbols_to_logits_fn, generation_initializer

    def call(self, inputs, is_training=True):
        symbols_to_logits_fn, generation_initializer = self.get_symbols_to_logits_fn(
            inputs, is_training=is_training, is_inference=False)

        text2text_logits = symbols_to_logits_fn(generation_initializer["decoder_input"],
                                                generation_initializer["text_decoder_internal_cache"])
        speech2text_logits = symbols_to_logits_fn(generation_initializer["decoder_input"],
                                                  generation_initializer["audio_decoder_internal_cache"])
        if generation_initializer["is_text2text"]:
            return text2text_logits
        else:
            return speech2text_logits
