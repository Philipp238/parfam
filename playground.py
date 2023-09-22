import tracemalloc

tracemalloc.start()

# Allocate some memory
a = [1] * (10 ** 6)

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

for stat in top_stats[:10]:
    print(stat)