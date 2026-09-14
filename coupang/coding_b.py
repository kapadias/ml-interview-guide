# Batch B.
import heapq, collections

# --- 11. Merge Intervals ---------------------------------------------------
def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out

# --- 12. Meeting Rooms II --------------------------------------------------
def min_meeting_rooms(intervals):
    if not intervals:
        return 0
    starts = sorted(i[0] for i in intervals)
    ends = sorted(i[1] for i in intervals)
    rooms = best = 0
    j = 0
    for s in starts:
        while j < len(ends) and ends[j] <= s:
            rooms -= 1
            j += 1
        rooms += 1
        best = max(best, rooms)
    return best

# --- 13. Course Schedule (topological sort) --------------------------------
def can_finish(num_courses, prerequisites):
    graph = collections.defaultdict(list)
    indeg = [0] * num_courses
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        indeg[course] += 1
    queue = collections.deque(c for c in range(num_courses) if indeg[c] == 0)
    seen = 0
    while queue:
        node = queue.popleft()
        seen += 1
        for nxt in graph[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    return seen == num_courses

# --- 14. Number of Islands -------------------------------------------------
def num_islands(grid):
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])
    seen = set()
    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != '1' or (r, c) in seen:
                continue
            count += 1
            stack = [(r, c)]                    # iterative: recursion overflows
            seen.add((r, c))
            while stack:
                y, x = stack.pop()
                for ny, nx in ((y+1,x), (y-1,x), (y,x+1), (y,x-1)):
                    if 0 <= ny < rows and 0 <= nx < cols \
                       and grid[ny][nx] == '1' and (ny, nx) not in seen:
                        seen.add((ny, nx))
                        stack.append((ny, nx))
    return count

# --- 15. Word Ladder (BFS shortest path) -----------------------------------
def ladder_length(begin, end, word_list):
    words = set(word_list)
    if end not in words:
        return 0
    queue = collections.deque([(begin, 1)])
    seen = {begin}
    while queue:
        word, steps = queue.popleft()
        if word == end:
            return steps
        for i in range(len(word)):
            for c in "abcdefghijklmnopqrstuvwxyz":
                nxt = word[:i] + c + word[i+1:]
                if nxt in words and nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, steps + 1))
    return 0

# --- 16. Coin Change -------------------------------------------------------
def coin_change(coins, amount):
    INF = amount + 1
    dp = [0] + [INF] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a:
                dp[a] = min(dp[a], dp[a - c] + 1)
    return -1 if dp[amount] == INF else dp[amount]

# --- 17. Longest Increasing Subsequence (patience sort) --------------------
import bisect
def length_of_lis(nums):
    tails = []
    for n in nums:
        i = bisect.bisect_left(tails, n)
        if i == len(tails):
            tails.append(n)
        else:
            tails[i] = n
    return len(tails)

# --- 18. LRU Cache ---------------------------------------------------------
class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.data = collections.OrderedDict()
    def get(self, key):
        if key not in self.data:
            return -1
        self.data.move_to_end(key)
        return self.data[key]
    def put(self, key, value):
        if key in self.data:
            self.data.move_to_end(key)
        self.data[key] = value
        if len(self.data) > self.cap:
            self.data.popitem(last=False)

# --- 19. Kth Smallest Element in a Sorted Matrix (binary search on value) --
def kth_smallest_matrix(matrix, k):
    n = len(matrix)
    def count_le(x):
        # walk from bottom-left: O(n) per probe
        cnt, r, c = 0, n - 1, 0
        while r >= 0 and c < n:
            if matrix[r][c] <= x:
                cnt += r + 1
                c += 1
            else:
                r -= 1
        return cnt
    lo, hi = matrix[0][0], matrix[n-1][n-1]
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if count_le(mid) < k:
            lo = mid + 1
        else:
            hi = mid
    return lo

# --- 20. Sort products by a composite key (stable, multi-key) --------------
def rank_products(products):
    """products: list of dicts with score, rating, price.
    Rank by score desc, then rating desc, then price asc, then id asc."""
    return sorted(products,
                  key=lambda p: (-p["score"], -p["rating"], p["price"], p["id"]))
