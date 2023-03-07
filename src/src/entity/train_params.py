from dataclasses import dataclass, field

@dataclass
class TrainParams:
    """Structure for data parameters"""
    clip: int = field(default=1)
    num_epochs: int = field(default=30)
    teacher_forcing_ratio: float = field(default=0.35)
    translated_examples_file: str = field(default="./reports/baseline_translated_exams.log")