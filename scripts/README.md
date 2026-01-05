# Scripts

This folder holds helper scripts for working with the Luogu captcha CRNN.

Current scripts:

- `smoke_model.py`: quick forward/decode sanity check on a random tensor.
- `train.py`: train the CRNN on generated captchas with CTC loss.
- `predict.py`: fetch captchas from Luogu endpoint, predict, and display with matplotlib.

Feel free to add training, evaluation, and inference utilities here.
