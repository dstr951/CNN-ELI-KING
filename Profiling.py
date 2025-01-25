#To create file run python -m cProfile -o <name>.prof Main.py

import pstats
stats = pstats.Stats("profile_optimized_cnn.prof")
# stats.sort_stats("time").print_stats(10)
stats.print_callers("BatchNorm.py:28")