from seq2seq_attention import seq2seq_attention
from base_seq2seq import base_seq2seq


def get_model(model_type: str, logger):
    if model_type == "baseline":
        return base_seq2seq
    elif model_type == "seq2seq_attention":
        return seq2seq_attention
    else:
        logger.exception('Model name is incorrect')
        raise NotImplementedError()
