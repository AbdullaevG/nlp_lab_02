import sys
import logging

import pynndescent.distances

from data.make_dataset import iterators_and_fields
from entity.train_pipeline_params import read_training_pipeline_params
from models.base_seq2seq import base_seq2seq
from models.seq2seq_attention import seq2seq_attention
import click
from train_model import train_model
import torch
import torch.nn as nn
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(model_type: str):
    if model_type == "baseline":
        return base_seq2seq
    elif model_type == "seq2seq_attention":
        return seq2seq_attention
    else:
        logger.exception('Model name is incorrect')
        raise NotImplementedError()


def train_pipeline(config_path: str, report_file: str, translated_exams_file: str):
    """train pipeline"""
    all_params = read_training_pipeline_params(config_path)

    data_params_dict = vars(all_params.dataparams)
    data, fields = iterators_and_fields(**data_params_dict)
    train_iterator, valid_iterator, test_iterator = data
    SRC, TRG = fields

    model_params = vars(all_params.seq2seqparams)
    logger.info("Try build model...")
    model = get_model(model_params["model_type"])
    model, save_model_path = model(len(SRC.vocab), len(TRG.vocab), **model_params)
    logger.info("baseline loaded!!!")
    PAD_IDX = TRG.vocab.stoi['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters())
    train_params = all_params.trainparams
    train_model(model,
                optimizer,
                criterion,
                train_iterator,
                valid_iterator,
                TRG.vocab,
                report_file,
                save_model_path,
                train_params.clip,
                train_params.num_epochs,
                train_params.teacher_forcing_ratio,
                translated_exams_file,
                logger)

@click.command(name='train_pipeline')
@click.argument('config_path', default='configs/train_config_seq2seq_attention.yml')
@click.argument('report_file', default='reports/seq2seq_attention.log')
@click.argument('translated_exams_file', default='reports/seq2seq_attention_generated.txt')
def train_pipeline_command(config_path: str, report_file: str, translated_exams_file: str):
    """ Make start for terminal """
    train_pipeline(config_path, report_file, translated_exams_file)


if __name__ == '__main__':
    train_pipeline_command()