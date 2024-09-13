import os
secret_key = os.urandom(24)  # Generates a 24-byte (192-bit) random key
print(secret_key)
