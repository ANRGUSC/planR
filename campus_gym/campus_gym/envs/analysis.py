import itertools
import random
if __name__ == '__main__':
    x = [0, 1, 2]
    all_permutations = [p for p in itertools.product(x, repeat=3)]
    color_list = len(all_permutations)
    get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF), range(n)))
    permutation_colors = get_colors(color_list)
    print(permutation_colors)


