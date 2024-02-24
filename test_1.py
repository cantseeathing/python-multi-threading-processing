import functools
import threading
import os
import time
from functools import wraps
import multiprocessing
import numpy
import numpy as np
import random
import concurrent.futures

# lock = threading.Lock()

random.seed(0)
np.random.seed(0)


def time_checker(func):
    @wraps(func)
    def wrapper_function(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print('****************************************')
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        print('****************************************')
        return result

    return wrapper_function


@time_checker
def task1():
    print("Task 1 assigned to thread: {}".format(threading.current_thread().name))
    time.sleep(10)
    print("ID of process running task 1: {}".format(os.getpid()))


@time_checker
def task2():
    print("Task 2 assigned to thread: {}".format(threading.current_thread().name))
    time.sleep(10)
    print("ID of process running task 2: {}".format(os.getpid()))


# print("ID of process running main program: {}".format(os.getpid()))
#
# print("Main thread name: {}".format(threading.current_thread().name))
#
# t1 = threading.Thread(target=task1, name='t1')
# t2 = threading.Thread(target=task2, name='t2')
#
# t1.start()
# t2.start()
#
#
# t1.join()
# t2.join()
# print('test')

# if __name__ == '__main__':
#     p1 = multiprocessing.Process(target=task1)
#     p2 = multiprocessing.Process(target=task2)
#
#     p1.start()
#     p2.start()
#
#     p1.join()
#     p2.join()
#
#     print('test')

arr = np.zeros(1_000_000, dtype='int')


@time_checker
def standard_test(object_test):
    for segment in range(0, object_test.size, object_test.size // 1_000):
        for items in object_test[segment:segment + object_test.size // 1000]:
            items = random.randint(0, 10) + random.randint(0, 10) * random.randint(0, 10) - random.randint(0, 10)


@time_checker
def thread_test(object_test):
    threads = []

    def thread_task(seg):
        for items in object_test[seg:seg + object_test.size // 1000]:
            items = random.randint(0, 10) + random.randint(0, 10) * random.randint(0, 10) - random.randint(0, 10)

    for segment in range(0, object_test.size, object_test.size // 1_000):
        t = threading.Thread(target=thread_task, args=[segment])
        t.start()
        threads.append(t)

    for thread in threads:
        thread.join()


def process_task(seg, object_test):
    for items in object_test[seg:seg + object_test.size // 1000]:
        items = random.randint(0, 10) + random.randint(0, 10) * random.randint(0, 10) - random.randint(0, 10)


def process_task_v2(seg):
    for items in arr[seg:seg + arr.size // 1000]:
        items = random.randint(0, 10) + random.randint(0, 10) * random.randint(0, 10) - random.randint(0, 10)


def thread_task_v2(seg):
    for items in arr[seg:seg + arr.size // 1000]:
        items = random.randint(0, 10) + random.randint(0, 10) * random.randint(0, 10) - random.randint(0, 10)


@time_checker
def multiprocess_test(object_test):
    processes = []

    for segment in range(0, object_test.size, object_test.size // 1_000):
        p = multiprocessing.Process(target=process_task, args=[segment, object_test])
        p.start()
        processes.append(p)

    for process in processes:
        process.join()


@time_checker
def multiprocess_test_v2():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process_task, [x for x in range(0, arr.size, arr.size // 1_000)])


@time_checker
def thread_test_v2():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(thread_task_v2, [x for x in range(0, arr.size, arr.size // 1_000)])


if __name__ == "__main__":
    standard_test(arr)
    thread_test(arr)
    multiprocess_test(arr)
    multiprocess_test_v2()
    thread_test_v2()
