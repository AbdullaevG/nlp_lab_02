import torch
import torch.nn as nn
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.p = dropout
        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=emb_dim
        )

        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )

        self.dropout = nn.Dropout(p=self.p)

    def forward(self, src):
        # src = [src_len, batch_size]

        embedded = self.embedding(src)
        embedded = self.dropout(embedded)
        # embedded = [src_len, batch_size, emb_dim]

        output, (hidden, cell) = self.rnn(embedded)
        # outputs = [src_len, batch size, hid_dim * n_directions]
        # hidden = [n_layers * n_directions, batch_size, hid_dim]
        # cell = [n_layers * n_directions, batch_size, hid_dim]

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(
            num_embeddings=self.output_dim,
            embedding_dim=emb_dim
        )

        self.rnn = nn.LSTM(
            input_size=self.emb_dim,
            hidden_size=self.hid_dim,
            num_layers=self.n_layers,
            dropout=dropout
        )

        self.fc = nn.Linear(self.hid_dim, self.output_dim)

    def forward(self, input_, hidden, cell):
        # input_=[batch_size], hidden, cell=[num_layers*num_directions, batch_size, hid_size]
        input_ = input_.unsqueeze(0)  # input_=[1, batch_size]
        embedding = self.embedding(input_)
        # embedding = [1, batch_size, emb_dim]
        output, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # output = [1, batch_size, hid_size], hidden,= [num_layers*num_directions, batch_size, hid_size]
        logits = self.fc(output.squeeze(0))
        # logits = [batch_size, trg_vocab_len]
        return logits, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device=device):
        super().__init__()

        self.enc = encoder
        self.dec = decoder
        self.device = device


    def forward(self, src, trg, teacher_forcing_ratio):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.dec.output_dim
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        h_pred, c_pred = self.enc(src)

        input_ = trg[0, :]  # first token is <sos>
        for t in range(1, max_len):
            # print(max(input_))
            dec_output_t, h_pred, c_pred = self.dec(input_, h_pred, c_pred)
            outputs[t] = dec_output_t
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = dec_output_t.max(1)[1]
            input_ = (trg[t] if teacher_force else top1)

        return outputs


def init_weights(model):
    for name, param in model.named_parameters():
        nn.init.uniform_(param, -0.08, 0.08)


def base_seq2seq(model_type,
                 enc_inp_dim,
                 dec_out_dim,
                 enc_emb_dim,
                 dec_emb_dim,
                 enc_hid_dim,
                 dec_hid_dim,
                 enc_dropout,
                 dec_dropout,
                 n_layers,
                 save_path):
    encoder = Encoder(enc_inp_dim, enc_emb_dim, enc_hid_dim, n_layers, enc_dropout)
    decoder = Decoder(dec_out_dim, dec_emb_dim, dec_hid_dim, n_layers, dec_dropout)
    seq2seq = Seq2Seq(encoder, decoder)
    seq2seq.apply(init_weights)
    seq2seq.to(device)

    return seq2seq, save_path