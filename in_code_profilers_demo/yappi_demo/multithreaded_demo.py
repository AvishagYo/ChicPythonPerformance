import yappi
import time
import threading

_NTHREAD = 3


def _work(n):
    time.sleep(n * 0.1)


yappi.start()

threads = []
# generate _NTHREAD threads
for i in range(_NTHREAD):
    t = threading.Thread(target=_work, args=(i + 1, ))
    t.start()
    threads.append(t)
# wait all threads to finish
for t in threads:
    t.join()

yappi.stop()

# retrieve thread stats by their thread id (given by yappi)
threads = yappi.get_thread_stats()
for thread in threads:
    print(
        "Function stats for (%s) (%d)" % (thread.name, thread.id)
    )  # it is the Thread.__class__.__name__
    yappi.get_func_stats(ctx_id=thread.id).print_all()
