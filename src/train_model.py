import time
import tqdm
import logging
import numpy as np
import torch
from utils import get_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def step(model,
         optimizer,
         criterion,
         clip,
         iterator,
         trg_vocab,
         teacher_forcing_ratio,
         phase="train",
         device=device):
    if phase == "train":
        model.train()
        tfr = teacher_forcing_ratio
    else:
        model.eval()
        tfr = 0

    epoch_loss = 0
    start_time = time.time()

    for i, batch in tqdm.tqdm(enumerate(iterator)):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg, tfr)
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        if phase == "train":
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        epoch_loss += loss.cpu().item()

    end_time = time.time()
    return epoch_loss / len(iterator), end_time - start_time


def train_model(model,
                optimizer,
                criterion,
                train_iterator,
                valid_iterator,
                trg_vocab,
                logging_file,
                best_model_path,
                clip,
                num_epochs,
                teacher_forcing_ratio,
                translated_examples_file,
                logger):
    best_loss = np.inf
    start_train = time.time()

    f_handler = logging.FileHandler(logging_file)
    f_handler.setLevel(logging.ERROR)
    f_format = logging.Formatter('%(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
    model.to(device)
    logger.info("Start training model... \n")
    for num_epoch in range(num_epochs):
        logger.info(f"epoch: {num_epoch + 1}")
        train_loss, train_time = step(model, optimizer, criterion, clip, train_iterator, trg_vocab,
                                      teacher_forcing_ratio, phase="train")
        valid_loss, valid_time = step(model, optimizer, criterion, clip, valid_iterator, trg_vocab, 0, phase="valid")

        if valid_loss < best_loss:
            best_ppl = np.exp(valid_loss)
            torch.save(model.state_dict(), best_model_path)

        logger.info(
            f"train time: {(train_time // 60):.0f} m {(train_time % 60):.0f} s, loss: {train_loss:.3f}, PPL: {np.exp(train_loss):.3f}")
        logger.info(
            f"valid time: {(valid_time // 60):.0f} m {(valid_time % 60):.0f} s, loss: {valid_loss:.3f}, PPL: {np.exp(valid_loss):.3f}\n")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    blue_score = get_bleu(model, valid_iterator, trg_vocab, translated_examples_file)
    end_train = time.time()
    time_elapsed = start_train - end_train
    logger.info(f"Training complete in {(time_elapsed // 60):.0f} m {(time_elapsed % 60):.0f} s")
    logger.info(f"Model state saved at {best_model_path}, the best bleu score is: {blue_score:.3f}, PPL: {best_ppl:.3f}")


def validate_best_model(model, best_model_path, valid_iterator, trg_vocab, translated_examples_file):
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    blue_score = get_bleu(model, valid_iterator, trg_vocab, translated_examples_file)
