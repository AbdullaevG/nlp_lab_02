import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, num_layers=self.n_layers, bidirectional=True)
        self.fc_hid = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.fc_cell = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp):
        # src = [src_len, batch_size]
        embedded = self.dropout(self.embedding(inp))

        # embedded = [src len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs = [src_len, batch_size, hid_dim * num_directions]
        # hidden = [n_layers * num directions, batch_size, hid_dim]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc_hid(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        cell = torch.tanh(self.fc_cell(torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)))
        # outputs = [src_len, batch_size, enc_hid_dim * 2] - concat hid vector for every token
        # hidden = [batch_size, dec_hid_dim] - changed with fc hidden state from last step

        return outputs, (hidden, cell)


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear(2 * enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # torch.Size([128, 32]) torch.Size([39, 128, 64])

        # hidden = [batch_size, dec_hid_dim] - hidden state from previous state of decoder
        # encoder_outputs = [src_len, batch_size, enc_hid_dim * 2] - all hidden states from encoder

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # hidden = [batch_size, src_len, dec_hid_dim]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch_size, src_len, enc_hid_dim * 2]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch_size, src_len, dec_hid_dim]

        attention = self.v(energy).squeeze(2)
        # attention= [batch_size, src_len]

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        # input = [batch size]
        # hidden = [batch_size, dec_hid_dim]
        # encoder_outputs = [src_len, batch_size, enc_hid_dim * 2]

        input = input.unsqueeze(0)
        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch_size, emb_dim]

        a = self.attention(hidden, encoder_outputs)
        # a = [batch_size, src_len]

        a = a.unsqueeze(1)
        # a = [batch_size, 1, src_len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch_size, src_len, enc_hid_dim * 2]

        weighted = torch.bmm(a, encoder_outputs)
        # weighted = [batch_size, 1, enc_hid_dim * 2]

        weighted = weighted.permute(1, 0, 2)
        # weighted = [1, batch_size, enc_hid_dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input = [1, batch_size, (enc_hid_dim * 2) + emb_dim]

        output, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))
        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # seq_len, n_layers and n_directions will always be 1 in this decoder, therefore:
        # output = [1, batch_size, dec_hid_dim]
        # hidden = [1, batch_size, dec_hid_dim]
        # this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        # prediction = [batch size, output dim]

        return prediction, (hidden.squeeze(0), cell.squeeze(0))


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer

        encoder_outputs, (hidden, cell) = self.encoder(src)
        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, (hidden, cell) = self.decoder(input, hidden, cell, encoder_outputs)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def seq2seq_attention(input_dim,
                      output_dim,
                      enc_emb_dim,
                      dec_emb_dim,
                      enc_hid_dim,
                      dec_hid_dim,
                      enc_dropout,
                      dec_dropout,
                      n_layers,
                      save_path,
                      model_type):
    attn = Attention(enc_hid_dim, dec_hid_dim)
    enc = Encoder(input_dim, enc_emb_dim, enc_hid_dim, dec_hid_dim, n_layers, enc_dropout)
    dec = Decoder(output_dim, dec_emb_dim, enc_hid_dim, dec_hid_dim, dec_dropout, attn)

    model = Seq2Seq(enc, dec, device)
    model.apply(init_weights).to(device)

    return model, save_path


def main():
    pass


if __name__ == "__main__":
    main()