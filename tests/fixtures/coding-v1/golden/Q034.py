def run_with_retry(operation):
    last_error = None
    for _ in range(3):
        try:
            return operation()
        except Exception as error:
            last_error = error
    raise last_error
