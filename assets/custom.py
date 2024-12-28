import tiktoken
import struct

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("As a large language model, I")

print(len(tokens))

with open("custom", "wb") as f:
    for token in tokens:
        f.write(struct.pack("H", token))
