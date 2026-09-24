def is_palindrome(text):
    chars = [char.lower() for char in text if char.isalnum()]
    return chars == chars[::-1]
