import multiprocessing
import tqdm 
def dummy(x):
    return x

if __name__ == '__main__':

    with multiprocessing.Pool(processes=4) as pool:
        r = list(tqdm.tqdm(pool.imap(dummy, range(100)), total=100))