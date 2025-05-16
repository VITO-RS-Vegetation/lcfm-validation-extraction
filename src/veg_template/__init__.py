import concurrent.futures
import random
import string

from tqdm.auto import tqdm

__all__ = ["__version__"]


def parallelize(
    f, my_iter, max_workers=4, progressbar=True, total=None, use_process_pool=False
):
    if total is None:
        try:
            total = len(my_iter)
        except Exception:
            total = None
            progressbar = False

    if use_process_pool:
        Pool = concurrent.futures.ProcessPoolExecutor
    else:
        Pool = concurrent.futures.ThreadPoolExecutor

    with Pool(max_workers=max_workers) as ex:
        if progressbar:
            results = list(tqdm(ex.map(f, my_iter), total=total))
        else:
            results = list(ex.map(f, my_iter))
    return results


def random_string(n=8):
    x = "".join(
        random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits)
        for _ in range(n)
    )
    return x
