#!/usr/bin/env python3
"""Exercise every reference solution shipped in the coding section.

Run with `make coupang-test`. A prep document containing a subtly wrong solution
is worse than one containing no code, so nothing ships unexercised.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coding_a import *          # noqa: F401,F403
from coding_b import *          # noqa: F401,F403


def check(name, got, want):
    if got != want:
        print("FAIL %-34s got %r want %r" % (name, got, want))
        return 0
    return 1


def main():
    p = 0
    t = 0

    def c(name, got, want):
        nonlocal p, t
        t += 1
        p += check(name, got, want)

    c("top_k_frequent", sorted(top_k_frequent([1, 1, 1, 2, 2, 3], 2)), [1, 2])
    c("top_k_frequent single", top_k_frequent([1], 1), [1])
    c("merge_k_sorted", merge_k_sorted([[1, 4, 5], [1, 3, 4], [2, 6]]),
      [1, 1, 2, 3, 4, 4, 5, 6])
    c("merge_k_sorted empties", merge_k_sorted([[], [1], []]), [1])
    c("merge_k_sorted all empty", merge_k_sorted([[], []]), [])

    k = KthLargest(3, [4, 5, 8, 2])
    c("KthLargest", [k.add(3), k.add(5), k.add(10), k.add(9), k.add(4)],
      [4, 5, 5, 8, 8])
    c("KthLargest short init", KthLargest(2, [1]).add(5), 1)

    tr = Trie()
    tr.insert("apple")
    c("Trie word", tr.search("apple"), True)
    c("Trie prefix is not word", tr.search("app"), False)
    c("Trie startsWith", tr.starts_with("app"), True)
    c("Trie missing", tr.starts_with("b"), False)

    c("suggested_products",
      suggested_products(["mobile", "mouse", "moneypot", "monitor", "mousepad"],
                         "mouse"),
      [["mobile", "moneypot", "monitor"], ["mobile", "moneypot", "monitor"],
       ["mouse", "mousepad"], ["mouse", "mousepad"], ["mouse", "mousepad"]])
    c("suggested_products no match",
      suggested_products(["aaa"], "b"), [[]])

    c("edit_distance", edit_distance("horse", "ros"), 3)
    c("edit_distance empty", edit_distance("", "abc"), 3)
    c("edit_distance identical", edit_distance("a", "a"), 0)
    c("edit_distance both empty", edit_distance("", ""), 0)

    c("group_anagrams",
      sorted(sorted(g) for g in group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])),
      sorted(sorted(g) for g in [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]))
    c("group_anagrams empty string", group_anagrams([""]), [[""]])

    c("longest_unique", longest_unique("abcabcbb"), 3)
    c("longest_unique all same", longest_unique("bbbbb"), 1)
    c("longest_unique empty", longest_unique(""), 0)
    c("longest_unique pwwkew", longest_unique("pwwkew"), 3)
    c("longest_unique stale index", longest_unique("abba"), 2)

    c("min_window", min_window("ADOBECODEBANC", "ABC"), "BANC")
    c("min_window impossible", min_window("a", "aa"), "")
    c("min_window exact", min_window("a", "a"), "a")
    c("min_window empty", min_window("", "a"), "")

    c("search_rotated found", search_rotated([4, 5, 6, 7, 0, 1, 2], 0), 4)
    c("search_rotated absent", search_rotated([4, 5, 6, 7, 0, 1, 2], 3), -1)
    c("search_rotated single", search_rotated([1], 0), -1)
    c("search_rotated unrotated", search_rotated([1, 2, 3], 3), 2)

    c("merge_intervals", merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]),
      [[1, 6], [8, 10], [15, 18]])
    c("merge_intervals touching", merge_intervals([[1, 4], [4, 5]]), [[1, 5]])
    c("merge_intervals empty", merge_intervals([]), [])
    c("merge_intervals nested", merge_intervals([[1, 10], [2, 3]]), [[1, 10]])

    c("min_meeting_rooms", min_meeting_rooms([[0, 30], [5, 10], [15, 20]]), 2)
    c("min_meeting_rooms disjoint", min_meeting_rooms([[7, 10], [2, 4]]), 1)
    c("min_meeting_rooms empty", min_meeting_rooms([]), 0)
    c("min_meeting_rooms touching", min_meeting_rooms([[1, 5], [5, 10]]), 1)

    c("can_finish ok", can_finish(2, [[1, 0]]), True)
    c("can_finish cycle", can_finish(2, [[1, 0], [0, 1]]), False)
    c("can_finish no prereqs", can_finish(1, []), True)
    c("can_finish self loop", can_finish(1, [[0, 0]]), False)

    g1 = [list("11110"), list("11010"), list("11000"), list("00000")]
    c("num_islands one", num_islands(g1), 1)
    g2 = [list("11000"), list("11000"), list("00100"), list("00011")]
    c("num_islands three", num_islands(g2), 3)
    c("num_islands empty", num_islands([]), 0)
    c("num_islands no land", num_islands([list("000")]), 0)

    c("ladder_length", ladder_length("hit", "cog",
      ["hot", "dot", "dog", "lot", "log", "cog"]), 5)
    c("ladder_length no end", ladder_length("hit", "cog",
      ["hot", "dot", "dog", "lot", "log"]), 0)

    c("coin_change", coin_change([1, 2, 5], 11), 3)
    c("coin_change impossible", coin_change([2], 3), -1)
    c("coin_change zero", coin_change([1], 0), 0)
    c("coin_change greedy trap", coin_change([1, 3, 4], 6), 2)

    c("length_of_lis", length_of_lis([10, 9, 2, 5, 3, 7, 101, 18]), 4)
    c("length_of_lis all equal", length_of_lis([7, 7, 7]), 1)
    c("length_of_lis empty", length_of_lis([]), 0)

    lru = LRUCache(2)
    lru.put(1, 1)
    lru.put(2, 2)
    got = [lru.get(1)]
    lru.put(3, 3)
    got.append(lru.get(2))
    lru.put(4, 4)
    got += [lru.get(1), lru.get(3), lru.get(4)]
    c("LRUCache", got, [1, -1, -1, 3, 4])
    lru2 = LRUCache(2)
    lru2.put(1, 1)
    lru2.put(1, 9)
    c("LRUCache update", lru2.get(1), 9)

    c("kth_smallest_matrix",
      kth_smallest_matrix([[1, 5, 9], [10, 11, 13], [12, 13, 15]], 8), 13)
    c("kth_smallest_matrix single", kth_smallest_matrix([[-5]], 1), -5)
    c("kth_smallest_matrix first",
      kth_smallest_matrix([[1, 2], [3, 4]], 1), 1)

    c("rank_products", [x["id"] for x in rank_products([
        {"id": 3, "score": 0.9, "rating": 4.5, "price": 100},
        {"id": 1, "score": 0.9, "rating": 4.5, "price": 80},
        {"id": 2, "score": 0.95, "rating": 3.0, "price": 200}])], [2, 1, 3])

    print("%d/%d pass" % (p, t))
    return 0 if p == t else 1


if __name__ == "__main__":
    sys.exit(main())
