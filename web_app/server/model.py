import tensorflow as tf
import tensorflow_addons as tfa
from encoder import Encoder
from decoder import Decoder
import utils
import pickle
import numpy as np

class Model:
    def __init__(self):
        self.decoder = None
        self.encoder = None
        self.optimizer = None
        self.en_tokenizer = None
        self.vi_tokenizer = None
        self.loaded_model = False
        self.batch_size = 1
        self.embedding_dim = 256
        self.units = 1024
        self.vocab_en_size = None
        self.vocab_vi_size = None
        self.attention = None
    
    def load_model(self, checkpoint_dir, dataset_file):
        with open(dataset_file, 'rb') as f:
            data = pickle.load(f)
            self.en_tokenizer = data['en_tokenizer']
            self.vi_tokenizer = data['vi_tokenizer']
            self.max_length_en = data['max_length_en']
            self.max_length_vi = data['max_length_vi']
            self.attention = data['attention']
            self.en_example = data['en_example']
            self.vi_example = data['vi_example']
            self.vocab_en_size = len(self.en_tokenizer.word_index)+1
            self.vocab_vi_size = len(self.vi_tokenizer.word_index)+1

        self.encoder = Encoder(self.vocab_en_size, self.embedding_dim, self.units, self.batch_size)
        self.decoder = Decoder(self.vocab_vi_size, self.embedding_dim, self.units, self.batch_size, self.max_length_en, self.max_length_vi, self.attention)
        self.optimizer = tf.keras.optimizers.Adam()
        
        self.re_train(tf.convert_to_tensor([self.en_example]), tf.convert_to_tensor([self.vi_example]))

        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                    encoder=self.encoder,
                                    decoder=self.decoder)

        status = checkpoint.restore(checkpoint_dir)
        self.loaded_model = True
        return status
    
    # BasicDecoder
    def evaluate_sentence(self, sentence):
        sentence = utils.preprocess_sentence(sentence)

        inputs = [self.en_tokenizer.word_index[i] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                                maxlen=self.max_length_en,
                                                                padding='post')
        inputs = tf.convert_to_tensor(inputs)
        inference_batch_size = inputs.shape[0]
        result = ''

        enc_start_state = [tf.zeros((inference_batch_size, self.units)), tf.zeros((inference_batch_size,self.units))]
        enc_out, enc_h, enc_c = self.encoder(inputs, enc_start_state)

        dec_h = enc_h
        dec_c = enc_c

        start_tokens = tf.fill([inference_batch_size], self.vi_tokenizer.word_index['<s>'])
        end_token = self.vi_tokenizer.word_index['</s>']

        greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

        decoder_instance = tfa.seq2seq.BasicDecoder(cell=self.decoder.rnn_cell, sampler=greedy_sampler, output_layer=self.decoder.fc)

        self.decoder.attention_mechanism.setup_memory(enc_out)

        decoder_initial_state = self.decoder.build_initial_state(inference_batch_size, [enc_h, enc_c], tf.float32)

        decoder_embedding_matrix = self.decoder.embedding.variables[0]

        outputs, _, _ = decoder_instance(decoder_embedding_matrix, start_tokens = start_tokens, end_token = end_token, initial_state=decoder_initial_state)
        return outputs.sample_id.numpy()

    def basic_translate(self, sentence):
        result = self.evaluate_sentence(sentence)
        result = self.vi_tokenizer.sequences_to_texts(result)
        return result


    # BeamSearchDecoder
    def beam_evaluate_sentence(self, sentence, beam_width=5):
        sentence = utils.preprocess_sentence(sentence)

        inputs = [self.en_tokenizer.word_index[i] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                                maxlen=self.max_length_en,
                                                                padding='post')
        inputs = tf.convert_to_tensor(inputs)
        inference_batch_size = inputs.shape[0]
        result = ''

        enc_start_state = [tf.zeros((inference_batch_size, self.units)), tf.zeros((inference_batch_size, self.units))]
        enc_out, enc_h, enc_c = self.encoder(inputs, enc_start_state)

        dec_h = enc_h
        dec_c = enc_c

        start_tokens = tf.fill([inference_batch_size], self.vi_tokenizer.word_index['<s>'])
        end_token = self.vi_tokenizer.word_index['</s>']

        enc_out = tfa.seq2seq.tile_batch(enc_out, multiplier=beam_width)
        self.decoder.attention_mechanism.setup_memory(enc_out)

        hidden_state = tfa.seq2seq.tile_batch([enc_h, enc_c], multiplier=beam_width)
        decoder_initial_state = self.decoder.rnn_cell.get_initial_state(batch_size=beam_width*inference_batch_size, dtype=tf.float32)
        decoder_initial_state = decoder_initial_state.clone(cell_state=hidden_state)

        decoder_instance = tfa.seq2seq.BeamSearchDecoder(self.decoder.rnn_cell,beam_width=beam_width, output_layer=self.decoder.fc)
        decoder_embedding_matrix = self.decoder.embedding.variables[0]

        outputs, final_state, sequence_lengths = decoder_instance(decoder_embedding_matrix, start_tokens=start_tokens, end_token=end_token, initial_state=decoder_initial_state)
        
        final_outputs = tf.transpose(outputs.predicted_ids, perm=(0,2,1))
        beam_scores = tf.transpose(outputs.beam_search_decoder_output.scores, perm=(0,2,1))

        return final_outputs.numpy(), beam_scores.numpy()

    def beam_translate(self, sentence, beam_width=5):
        print('beam_tran:', beam_width)
        result, beam_scores = self.beam_evaluate_sentence(sentence, beam_width=beam_width)
        
        for beam, score in zip(result, beam_scores):
            output = self.vi_tokenizer.sequences_to_texts(beam)
            output = [a[:a.index('</s>')] for a in output]
            output = [sent.replace('_', ' ') for sent in output]
            
            beam_score = [a.sum() for a in score]
            return output, beam_score

    def loss_function(self, real, pred):
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        loss = cross_entropy(y_true=real, y_pred=pred)
        mask = tf.logical_not(tf.math.equal(real,0))
        mask = tf.cast(mask, dtype=loss.dtype)  
        loss = mask* loss
        loss = tf.reduce_mean(loss)
        return loss

    @tf.function
    def train_step(self, inp, targ, enc_hidden):
        loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_h, enc_c = self.encoder(inp, enc_hidden)

            dec_input = targ[ : , :-1 ]
            real = targ[ : , 1: ]

            self.decoder.attention_mechanism.setup_memory(enc_output)

            decoder_initial_state = self.decoder.build_initial_state(self.batch_size, [enc_h, enc_c], tf.float32)
            pred = self.decoder(dec_input, decoder_initial_state)
            logits = pred.rnn_output
            loss = self.loss_function(real, logits)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

    def re_train(self, inp, targ):
        enc_hidden = self.encoder.initialize_hidden_state()
        batch_loss = self.train_step(inp, targ, enc_hidden)

    def translate(self, input_sentence):
        attention_plot = np.zeros((self.max_length_vi, self.max_length_en))
        sentence = utils.preprocess_sentence(input_sentence)

        inputs = [self.en_tokenizer.word_index[i] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                                maxlen=self.max_length_en,
                                                                padding='post')
        inputs = tf.convert_to_tensor(inputs)

        result = ''

        hidden = [tf.zeros((1, self.units))]
        enc_out, enc_hidden = self.encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.vi_tokenizer.word_index['<s>']], 0)

        for t in range(self.max_length_vi):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input,
                                                                dec_hidden,
                                                                enc_out)

            attention_weights = tf.reshape(attention_weights, (-1, ))
            attention_plot[t] = attention_weights.numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()
            result += self.vi_tokenizer.index_word[predicted_id] + ' '

            if self.vi_tokenizer.index_word[predicted_id] == '</s>':
                return result, sentence, attention_plot

            dec_input = tf.expand_dims([predicted_id], 0)
        return result, sentence, attention_plot