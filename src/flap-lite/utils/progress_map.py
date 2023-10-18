import time
import os
from tqdm.contrib.concurrent import process_map


def _test_func(x):
    time.sleep(0.01)
    return x + 2


def calc_chunksize(n_workers, len_iterable, factor=4):
    """Calculate chunksize argument for Pool-methods.

    Resembles source-code within `multiprocessing.pool.Pool._map_async`.
    """
    chunksize, extra = divmod(len_iterable, n_workers * factor)
    if extra:
        chunksize += 1
    return chunksize


def _progress_map_naive(func, iterable):
    map_result = []

    n = len(iterable)
    n_parts = min(20, n)
    n_tile = int(n / n_parts)
    tile_counter = 0
    counter = 0
    counter_c = 0
    t0 = time.time()
    tc = time.time()
    refresh_interval = 1

    for i in iterable:

        map_result.append(func(i))

        counter += 1
        if counter % n_tile == 0:
            tile_counter += 1

        if (time.time() - tc) > refresh_interval:
            progress = '+' * tile_counter + '>' + '-' * (n_parts - tile_counter - 1)
            print('\r[%s](%s/%s) [%.4f iter/s, time elapsed = %.2f s]' % (
                progress, counter, n, (counter - counter_c) / (time.time() - tc), time.time() - t0), end='\r')
            counter_c = counter
            tc = time.time()

    tile_counter += 1
    progress = '+' * tile_counter
    print('\r[%s](%s/%s) [%.4f s, total time = %.2f s]' % (
        progress, counter, n, counter / (time.time() - t0), time.time() - t0), end='\r')

    return map_result


def _progress_map_tqdm(func, iterable):
    from tqdm import tqdm
    map_result = [func(i) for i in tqdm(iterable)]
    return map_result


def progress_map_tqdm_concurrent(func, iterable, length=None, max_workers=None, chunksize=None):

    if max_workers is None:
        max_workers = os.cpu_count()

    if chunksize is None:
        if length is not None:
            chunksize = calc_chunksize(max_workers, length)
        else:
            try:
                chunksize = calc_chunksize(max_workers, len(iterable))
            except TypeError:
                chunksize = calc_chunksize(max_workers, 5000)

    map_result = process_map(func, iterable, max_workers=max_workers,
                             chunksize=chunksize)

    return map_result


def progress_map(func, iterable):

    try:
        map_result = _progress_map_tqdm(func, iterable)
    except ModuleNotFoundError:
        map_result = _progress_map_naive(func, iterable)

    return map_result
