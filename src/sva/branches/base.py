from typing import Protocol

from sva.schema import Evidence, Sample


class Branch(Protocol):
    name: str

    def run(self, sample: Sample) -> list[Evidence]:
        """Moi nhanh tra ve cac Evidence (phan phoi tren dung option cua mau), khong tra chuoi chu."""
        ...
