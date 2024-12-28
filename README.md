# gpt-2

This is my reimplementation of the forward and backward pass of GPT-2 in C, written in standard C11 with no dependencies other than the C POSIX library. My only intention with this project was to gain a low-level understanding of how transformer models work. The code is not production-ready.

To compile and run, execute `build.sh`. For my reference implementation in Python, see `ref.py`. I used this to verify the results I got in C, and it was originally compared against Andrej Karpathy's `nanoGPT` implementation. For both implementations, you must have a `model.safetensors` file in the `assets` directory. This file contains the parameter values of the model and can be downloaded from [HuggingFace](https://huggingface.co/openai-community/gpt2?show_file_info=model.safetensors).
