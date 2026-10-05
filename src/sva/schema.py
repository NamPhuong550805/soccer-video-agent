from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

ClaimType = Literal["categorical", "ordinal", "positional", "text", "failure"]


@dataclass
class Sample:
    task_id: str
    index: int                     # index trong file json goc
    question: str
    materials: list[str]           # duong dan tuyet doi
    option_keys: list[str]         # co the khong lien tuc, vd ["O1", "O3"]
    option_texts: dict[str, str]
    gold_key: str
    open_answer: Optional[str] = None
    cleaning_note: Optional[str] = None   # "closeA_fixed" | "openA_empty"
    meta: dict = field(default_factory=dict)

    @property
    def n_options(self):
        return len(self.option_keys)

    def to_json(self):
        return asdict(self)

    @classmethod
    def from_json(cls, d):
        return cls(**d)


@dataclass
class Evidence:
    """Ket qua cua mot tool: phan phoi tren dung cac option cua mau, khong phai mot chuoi."""
    tool_name: str
    claim_type: ClaimType
    option_keys: list[str]
    dist: dict[str, float]          # tong = 1, chi tren option_keys
    confidence: float               # p(top1) - p(top2)
    raw_observation: str = ""
    material_ref: dict = field(default_factory=dict)   # {"clips": [...], "timestamps": [...]}
    ordinal_scale: Optional[dict[str, float]] = None
    failed: bool = False

    def __post_init__(self):
        if set(self.dist) != set(self.option_keys):
            raise ValueError(f"dist {sorted(self.dist)} khong khop option_keys {self.option_keys}")
        if abs(sum(self.dist.values()) - 1.0) > 1e-6:
            raise ValueError(f"tong dist = {sum(self.dist.values())}")

    @property
    def pred(self):
        return max(self.option_keys, key=self.dist.get)

    @classmethod
    def failure(cls, tool_name, keys, reason):
        n = len(keys)
        return cls(tool_name, "failure", list(keys), {k: 1 / n for k in keys}, 0.0, reason, failed=True)

    def to_json(self):
        return asdict(self)
