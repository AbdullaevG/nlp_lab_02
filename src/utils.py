import torch
import time
import tqdm
from nltk.translate.bleu_score import corpus_bleu
import logging

def flatten(l):
    return [item for sublist in l for item in sublist]


def remove_tech_tokens(mystr, tokens_to_remove=['<eos>', '<sos>', '<unk>', '<pad>']):
    return [x for x in mystr if x not in tokens_to_remove]


def get_text(x, TRG_vocab):
    text = [TRG_vocab.itos[token] for token in x]
    try:
        end_idx = text.index('<eos>')
        text = text[:end_idx]
    except ValueError:
        pass
    text = remove_tech_tokens(text)
    if len(text) < 1:
        text = []
    return text


def get_bleu(model, test_iterator, trg_vocab, logging_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    original_text = []
    generated_text = []
    model.eval()
    logging.basicConfig(filename=logging_file,
                        filemode='w',
                        format='%(message)s',
                        level=logging.INFO,
                        force=True)
    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(test_iterator)):
            src = batch.src
            trg = batch.trg
            output = model(src, trg, 0)
            output = output.argmax(dim=-1)

            original_text.extend([get_text(x, trg_vocab) for x in trg.cpu().numpy().T])
            generated_text.extend([get_text(x, trg_vocab) for x in output[1:].detach().cpu().numpy().T])

        score = corpus_bleu([[text] for text in original_text], generated_text) * 100
    i = 0
    for original, translated  in zip(original_text, generated_text):
        if i > 10:
            break
        else:
            logging.info(" ".join(original) + "\t" + " ".join(translated) + "\n")
        i += 1


    return score


def generate_translation(src, trg, model, TRG_vocab):
    model.eval()

    output = model(src, trg, 0) #turn off teacher forcing
    output = output.argmax(dim=-1).cpu().numpy()

    original = get_text(list(trg[:,0].cpu().numpy()), TRG_vocab)
    generated = get_text(list(output[1:, 0]), TRG_vocab)
    
    print('Original: {}'.format(' '.join(original)))
    print('Generated: {}'.format(' '.join(generated)))
    print()
