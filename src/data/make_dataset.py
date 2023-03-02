import logging
import sys
from nltk.tokenize import WordPunctTokenizer
from torchtext.data import Field, BucketIterator, TabularDataset
import torch

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def tokenize(data: str):
    """
    Take as input the string and return the list of tokens.
    """
    tokenizer = WordPunctTokenizer()
    return tokenizer.tokenize(data.lower())


def fields():
    """
    Return the tuple of  filds, for source and target languages
    """
    SRC = Field(tokenize=tokenize,
                init_token='<sos>',
                eos_token='<eos>',
                 lower = True)

    TRG = Field(tokenize=tokenize,
                init_token='<sos>',
                eos_token='<eos>',
                lower = True)
    
    return SRC, TRG


def get_data_fields(path_to_data,
                    file_format="tsv",
                    split_ratio=[0.8, 0.15, 0.05],
                    min_word_freq=3):
    """
    path_to_data: path to the file,
    format: format of the file,
    split_ratio: the split ration between train, valid and test data
    min_freq: minimum frequency for the words, which will be taken to the vocab
    return: tuple(train, valid, test), tuple(src_field, trg_field)
    """
    SRC, TRG = fields()
    logger.info('Loading dataset from %s...', path_to_data)
    dataset = TabularDataset(path=path_to_data,
                             format=file_format,
                             fields=[("trg", TRG), ("src", SRC)],
                             )
    logger.info('Loading from %s finished', path_to_data)
    
    logger.info('Split the data to train, valid and test...')
    train_data, test_data, valid_data = dataset.split(split_ratio=split_ratio)
    logger.info(f"Number of train, valid and test examples: {len(train_data.examples), len(valid_data.examples), len(test_data.examples)}")
    
    logger.info('Build the vocabularis...')
    SRC.build_vocab(train_data, min_freq=min_word_freq)
    TRG.build_vocab(train_data, min_freq=min_word_freq)
    logger.info("Vocabs is built!")
    logger.info(f"The number of unique tokens in source (ru) vocabulary: {len(SRC.vocab)}")
    logger.info(f"The number of unique tokens in target (en) vocabulary: {len(TRG.vocab)}")
    return (train_data, valid_data, test_data), (SRC, TRG)


def _len_sort_key(x):
    return len(x.src)


def iterators_and_fields(path_to_data,
                         file_format="tsv",
                         split_ratio=[0.8, 0.15, 0.05],
                         min_word_freq=3,
                         batch_size=128):
    """
    return: tuple(train, valid, test data iterators), tuple(src_field, trg_field)
    """

    data, fields = get_data_fields(path_to_data=path_to_data,
                                   file_format=file_format,
                                   split_ratio=split_ratio,
                                   min_word_freq=min_word_freq
                                   )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_data, valid_data, test_data = data[0], data[1], data[2]
    SRC, TRG = fields
    logger.info(f"Try to build iterators...")
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data,
                                                                           valid_data,
                                                                           test_data),
                                                                           batch_size = batch_size,
                                                                           sort_key=_len_sort_key,
                                                                           device = device
                                                                          )
    logger.info(f"Iterators prepared!!!")                                                                    
    
    return (train_iterator, valid_iterator, test_iterator), (SRC, TRG)

