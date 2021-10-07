from pydantic import BaseModel


class DataColumn(BaseModel):
    """To use multiple targets: create List[DataColumn].

    Can cross check with the Dataset to make sure it generates the connect
    number, type and shape of targets.

    Similar to tensorflow feature column.

    Want to use this for both tf and torch.
    """

    name: str
    shape: None
    dtype: None
    transforms: list

    def get_value(self, example):
        v = example[self.name]
        for t in self.transforms or []:
            v = t(v)
        return v
