from collections import Counter

def top_k_frequent(words, k):
    if k <= 0:
        return []
    counts = Counter(words)
    return [word for word, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:k]]
