# Batch A. Each function is the reference solution embedded in coding.json.
import heapq, collections, bisect

# --- 1. Top K Frequent Elements -------------------------------------------
def top_k_frequent(nums, k):
    counts = collections.Counter(nums)
    # heapq.nlargest is O(n log k); sorting everything would be O(n log n).
    return heapq.nlargest(k, counts, key=counts.get)

# --- 2. Merge K Sorted Lists (postings-list merge) -------------------------
def merge_k_sorted(lists):
    heap = [(lst[0], i, 0) for i, lst in enumerate(lists) if lst]
    heapq.heapify(heap)
    out = []
    while heap:
        val, li, idx = heapq.heappop(heap)
        out.append(val)
        if idx + 1 < len(lists[li]):
            heapq.heappush(heap, (lists[li][idx + 1], li, idx + 1))
    return out

# --- 3. Kth Largest in a Stream -------------------------------------------
class KthLargest:
    def __init__(self, k, nums):
        self.k = k
        self.heap = list(nums)
        heapq.heapify(self.heap)
        while len(self.heap) > k:
            heapq.heappop(self.heap)
    def add(self, val):
        heapq.heappush(self.heap, val)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        return self.heap[0]

# --- 4. Implement Trie -----------------------------------------------------
class Trie:
    def __init__(self):
        self.root = {}
    def insert(self, word):
        node = self.root
        for ch in word:
            node = node.setdefault(ch, {})
        node['$'] = True
    def search(self, word):
        node = self._walk(word)
        return node is not None and '$' in node
    def starts_with(self, prefix):
        return self._walk(prefix) is not None
    def _walk(self, s):
        node = self.root
        for ch in s:
            if ch not in node:
                return None
            node = node[ch]
        return node

# --- 5. Search Suggestions System (typeahead) ------------------------------
def suggested_products(products, search_word):
    products = sorted(products)
    out, lo = [], 0
    for i, ch in enumerate(search_word):
        prefix = search_word[:i + 1]
        lo = bisect.bisect_left(products, prefix, lo)
        hits = []
        for p in products[lo:lo + 3]:
            if p.startswith(prefix):
                hits.append(p)
        out.append(hits)
    return out

# --- 6. Edit Distance (spell correction) -----------------------------------
def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1,          # delete
                           cur[j - 1] + 1,       # insert
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

# --- 7. Group Anagrams -----------------------------------------------------
def group_anagrams(words):
    groups = collections.defaultdict(list)
    for w in words:
        key = tuple(sorted(w))
        groups[key].append(w)
    return list(groups.values())

# --- 8. Longest Substring Without Repeating Characters ---------------------
def longest_unique(s):
    last, start, best = {}, 0, 0
    for i, ch in enumerate(s):
        if ch in last and last[ch] >= start:
            start = last[ch] + 1
        last[ch] = i
        best = max(best, i - start + 1)
    return best

# --- 9. Minimum Window Substring -------------------------------------------
def min_window(s, t):
    if not s or not t:
        return ""
    need = collections.Counter(t)
    missing = len(t)
    best = (float('inf'), 0, 0)
    start = 0
    for end, ch in enumerate(s):
        if need[ch] > 0:
            missing -= 1
        need[ch] -= 1
        while missing == 0:
            if end - start + 1 < best[0]:
                best = (end - start + 1, start, end)
            need[s[start]] += 1
            if need[s[start]] > 0:
                missing += 1
            start += 1
    return "" if best[0] == float('inf') else s[best[1]:best[2] + 1]

# --- 10. Search in Rotated Sorted Array ------------------------------------
def search_rotated(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        if nums[lo] <= nums[mid]:               # left half is sorted
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:                                   # right half is sorted
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1
