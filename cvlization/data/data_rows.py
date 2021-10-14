class DataRows:
    def __getitem__(self, i: int) -> dict:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
