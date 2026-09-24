def merge_intervals(intervals):
    ordered = sorted((list(pair) for pair in intervals), key=lambda pair: (pair[0], pair[1]))
    merged = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        elif end > merged[-1][1]:
            merged[-1][1] = end
    return merged
