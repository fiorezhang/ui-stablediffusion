pyinstaller -F ui-stablediffusion.py --copy-metadata tqdm --copy-metadata regex --copy-metadata packaging --copy-metadata requests --copy-metadata filelock --copy-metadata numpy --copy-metadata tokenizers --add-data="C:\Users\DN27\anaconda3\envs\ovui\Lib\site-packages\transformers;transformers" --add-data="C:\Users\DN27\anaconda3\envs\ovui\Lib\site-packages\openvino;openvino"