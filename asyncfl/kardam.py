from asyncfl.server import Server


class Kardam(Server):
    def __init__(self, dataset: str, model_name: str) -> None:
        super().__init__(dataset, model_name)