"""Splitting data  params"""

from dataclasses import dataclass, field

@dataclass
class DataParams:
    """Structure for data parameters"""
    path_to_data: str = field(default="data/raw/data_small.txt")
    file_format: str = field(default="tsv")
    split_ratio: list = field(default_factory=[0.8, 0.15, 0.05])
    min_word_freq: int = field(default=3)
    batch_size: int = field(default=128)
