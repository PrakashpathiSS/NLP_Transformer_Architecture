import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Dropout, LayerNormalization, MultiHeadAttention


@tf.keras.utils.register_keras_serializable(package="SummarizationTransformer")
class PositionalEmbedding(tf.keras.layers.Layer):

    def __init__(self, vocab_size, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.d_model = d_model

        self.token_embedding = Embedding(vocab_size, d_model, mask_zero=True)
        self.position_embedding = Embedding(sequence_length, d_model)

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(0, length, 1)
        positions = self.position_embedding(positions)
        tokens = self.token_embedding(inputs)
        return tokens + positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "sequence_length": self.sequence_length,
            "d_model": self.d_model,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="SummarizationTransformer")
class TransformerEncoderBlock(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
        )
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(d_model),
        ])
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, x, training=False, attention_mask=None):
        attn_output = self.attention(
            query=x, key=x, value=x,
            attention_mask=attention_mask,
            training=training,
        )
        x = self.norm1(x + self.dropout1(attn_output, training=training))

        ffn_output = self.ffn(x, training=training)
        x = self.norm2(x + self.dropout2(ffn_output, training=training))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="SummarizationTransformer")
class TransformerDecoderBlock(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        self.self_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
        )
        self.cross_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
        )
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(d_model),
        ])
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.norm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def call(
        self,
        x,
        encoder_output,
        training=False,
        self_attention_mask=None,
        cross_attention_mask=None,
    ):
        attn1 = self.self_attention(
            query=x, key=x, value=x,
            attention_mask=self_attention_mask,
            training=training,
        )
        x = self.norm1(x + self.dropout1(attn1, training=training))

        attn2 = self.cross_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            attention_mask=cross_attention_mask,
            training=training,
        )
        x = self.norm2(x + self.dropout2(attn2, training=training))

        ffn_output = self.ffn(x, training=training)
        x = self.norm3(x + self.dropout3(ffn_output, training=training))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate,
        })
        return config


def causal_mask(length):
    return tf.linalg.band_part(
        tf.ones((length, length), dtype=tf.int32),
        -1,
        0
    )


@tf.keras.utils.register_keras_serializable(package="SummarizationTransformer")
class Transformer(tf.keras.Model):

    def __init__(
        self,
        input_vocab_size,
        summary_vocab_size,
        max_input_length,
        max_summary_length,
        d_model=64,
        num_heads=4,
        ff_dim=128,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_vocab_size = input_vocab_size
        self.summary_vocab_size = summary_vocab_size
        self.max_input_length = max_input_length
        self.max_summary_length = max_summary_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        self.encoder_embedding = PositionalEmbedding(
            input_vocab_size, max_input_length, d_model
        )
        self.decoder_embedding = PositionalEmbedding(
            summary_vocab_size, max_summary_length, d_model
        )
        self.encoder = TransformerEncoderBlock(
            d_model, num_heads, ff_dim, dropout
        )
        self.decoder = TransformerDecoderBlock(
            d_model, num_heads, ff_dim, dropout
        )
        self.output_layer = Dense(summary_vocab_size)

    def call(self, inputs, training=False):
        input_tokens, decoder_tokens = inputs

        encoder_x = self.encoder_embedding(input_tokens)
        decoder_x = self.decoder_embedding(decoder_tokens)

        encoder_padding = tf.cast(input_tokens != 0, tf.int32)[:, None, :]
        decoder_padding = tf.cast(decoder_tokens != 0, tf.int32)[:, None, :]

        encoder_mask = tf.cast(
            encoder_padding[:, :, None] * encoder_padding[:, None, :],
            tf.int32,
        )

        target_length = tf.shape(decoder_tokens)[1]
        cmask = causal_mask(target_length)[None, :, :]

        decoder_mask = cmask * tf.cast(
            decoder_padding[:, None, :],
            tf.int32,
        )

        cross_mask = tf.cast(
            encoder_padding[:, None, :],
            tf.int32,
        )

        encoder_output = self.encoder(
            encoder_x,
            training=training,
            attention_mask=encoder_mask,
        )

        decoder_output = self.decoder(
            decoder_x,
            encoder_output,
            training=training,
            self_attention_mask=decoder_mask,
            cross_attention_mask=cross_mask,
        )

        return self.output_layer(decoder_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_vocab_size": self.input_vocab_size,
            "summary_vocab_size": self.summary_vocab_size,
            "max_input_length": self.max_input_length,
            "max_summary_length": self.max_summary_length,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate,
        })
        return config
