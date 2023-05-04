
from typing import List
from multiprocessing import Pool, RLock, Process
from multiprocessing.pool import AsyncResult
from tqdm import tqdm
import time


def func1():
    # print('Starting function 1')
    time.sleep(5)
    # print('Done with function 1')
    return 'Hello world'


class PoolManager():

    def __init__(self, processes=None, initializer=None, initargs=(),
                 maxtasksperchild=None, context=None) -> None:
        self.pool = Pool(processes, initializer, initargs, maxtasksperchild)
        self.tasks = []
        self.active_tasks = []
        self.results = []
        self.num_processes = processes
        self.capacity = 1.0

    def add_task(self, func, args, required_capacity: float = 0.0):
        self.tasks.append((func, args, required_capacity))
    
    def run(self, pbar=None):
        while len(self.tasks):
            if self.capacity >= self.tasks[-1][2]:
                func, args, cap = self.tasks.pop()
                self.capacity -= cap
                self.active_tasks.append((self.pool.apply_async(func, args), cap))
            
            for index, (t, cap) in enumerate(self.active_tasks):
                if t.ready():
                    self.results.append(t)
                    self.active_tasks.pop(index)
                    self.capacity += cap
                    if pbar:
                        pbar.update(1)

    def get_results(self) -> List[AsyncResult]:
        return self.results




# class DynPool(Pool):
#     def __init__(self, processes=None, initializer=None, initargs=(),
#                  maxtasksperchild=None, context=None) -> None:
#         super().__init__(processes, initializer, initargs, maxtasksperchild, context)


if __name__ == '__main__':

    pool_size = 4

    # active_workers
    list_of_tasks = []
    list_of_tasks.append((func1, [], 1.0))
    list_of_tasks.append((func1, [], 0.25))
    list_of_tasks.append((func1, [], 0.25))
    list_of_tasks.append((func1, [], 0.25))
    list_of_tasks.append((func1, [], 0.5))
    list_of_tasks.append((func1, [], 0.5))
    list_of_tasks.reverse()


    # pool = Pool(pool_size, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
    pbar = tqdm(total=6, position=0, leave=None, desc='Total')
    pm = PoolManager(pool_size, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))

    pm.add_task(func1, [], 1.0)
    pm.add_task(func1, [], 0.25)
    pm.add_task(func1, [], 0.25)
    pm.add_task(func1, [], 0.25)
    pm.add_task(func1, [], 0.5)
    pm.add_task(func1, [], 0.5)

    pm.run(pbar=pbar)

    results = pm.get_results()

    for res in results:
        res : AsyncResult
        print(res.get())

    # num_active_workers = 0
    # results = []
    # active_tasks = []
    # capacity = 1.0
    # while len(list_of_tasks):
    #     # print(list_of_tasks[-1])
    #     if capacity >= list_of_tasks[-1][2]:
    #         func, args, cap = list_of_tasks.pop()
    #         capacity -= cap
    #         print(f'New capacity={capacity}')
    #         active_tasks.append((pool.apply_async(func, args), cap))
    #     # if num_active_workers <=2:
    #     #     active_tasks.append(pool.apply_async(list_of_tasks.pop()[0]))
    #     #     num_active_workers += 1
        
    #     for index, (t, cap) in enumerate(active_tasks):
    #         t : AsyncResult
    #         if t.ready():
    #             results.append(t)
    #             active_tasks.pop(index)
    #             capacity += cap
    #             # num_active_workers -= 1
        

    
    # pool_check(pool)
    # results.append(pool.apply_async(func1))
    # pool_check(pool)

    # results.append(pool.apply_async(func1))
    # pool_check(pool)

    # results.append(pool.apply_async(func1))
    # results.append(pool.apply_async(func1))
    # results.append(pool.apply_async(func1))
    # results.append(pool.apply_async(func1))
    # results.append(pool.apply_async(func1))
    # results.append(pool.apply_async(func1))
    # print(pool)
    # end = False
    # while not end:
    #     for res in results:
    #         res: AsyncResult
    #         end = all([x.ready() for x in results])
    #         # print(res.ready())

    print('Hello')