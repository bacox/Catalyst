from typing import Callable, Any, List


class Task:
    def __init__(self, entity, func: Callable, *args: List[Any]):
        self.t_id = 0
        self.entity = entity
        self.func = func
        self.args = args



    def __call__(self, *args, **kwargs):
        self.func(self.entity, *self.args)


    def __repr__(self):
        return f'[{self.entity}].{self.func}({self.args})'
