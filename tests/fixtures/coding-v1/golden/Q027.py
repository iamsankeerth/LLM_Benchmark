def dedupe_preserve_order(items):
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result
