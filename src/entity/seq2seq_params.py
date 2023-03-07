from dataclasses import dataclass, field

@dataclass
class Seq2SeqParams:
    """Structure for data parameters"""
    model_type: str = field(default = "seq2seq_attention")
    enc_emb_dim: int = field(default=4)
    dec_emb_dim: int = field(default=4)
    enc_hid_dim: int = field(default=8)
    dec_hid_dim: int = field(default=8)
    enc_dropout: float = field(default=0.15)
    dec_dropout: float = field(default=0.15)
    n_layers: int = field(default=1)
    save_path: str = field(default="./src/models/baseline.pt")
