"""Training model pipeline"""

from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml
from .data_params import DataParams
from .seq2seq_params import Seq2SeqParams
from .train_params import TrainParams


@dataclass
class TrainingPipelineParams:
    """Structure for pipeline parameters"""
    dataparams: DataParams
    seq2seqparams: Seq2SeqParams
    trainparams: TrainParams

TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str):
    """Read config for model training"""
    with open(path, 'r', encoding='utf-8') as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))

