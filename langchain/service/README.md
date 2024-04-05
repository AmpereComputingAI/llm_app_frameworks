Create model folder and download llama-2-7b-chat.Q4_K_M.gguf in it.

```
# mkdir model
# cd model
# wget "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
# cd ..
```

Create data folder and copy test document into it.

```
# mkdir data
# cp state_of_the_union.txt ./data
# cd ..
```

Run application

```
# python app.py
```

Launch a Web browser and load page http://localhost:8000/docs

