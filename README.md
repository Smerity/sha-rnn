# Single Headed Attention RNN

In summary, "stop thinking with your (attention) head".

- Obtain strong results on a byte level language modeling dataset (enwik8) in under 24 hours on a single GPU (12GB Titan V)
- Support long range dependencies (up to 5000 tokens) without increasing compute time or memory usage substantially by using a simpler attention mechanism
- Avoid the fragile training process required by standard Transformer models such as a long warmup
- Back off toward a standard LSTM allowing you to drop retained memory states (needed for a Transformer model) if memory becomes a major constraint
- Provide a smaller model that features only standard components such as the LSTM, single headed attention, and feed-forward modules such that they can easily be productionized using existing optimized tools and exported to various formats (i.e. ONNX)

For full details refer to [the SHA-RNN submission](https://devpost.com/submit-to/8320-global-pytorch-summer-hackathon/start/submissions/127170-single-headed-attention-rnn) on the PyTorch Global Hackathon page.

| Model                             | Test BPC | Params | LSTM Based |
|-----------------------------------|----------|--------|------------|
| Krause mLSTM                      | 1.24     | 46M    | ✔          |
| AWD-LSTM                          | 1.23    | 44M    | ✔          |
| **SHA-LSTM**                          | 1.07     | 63M    | ✔          |
| 12L Transformer-XL                | 1.06     | 41M    |            |
| 18L Transformer-XL                | 1.03     | 88M    |            |
| Adaptive Span Transformer (Small) | 1.02     | 38M    |            |

Whilst the model is still quite some way away from state of the art (~0.98 bpc) the model is low resource and high efficiency without having yet been optimized to be so.
The model was trained in only 24 hours on a single GPU with the [Adaptive Span Transformer](https://github.com/facebookresearch/adaptive-span) (small) being the only recent Transformer model to achieve the same type of training efficiency.

## To recreate

### Setup

To get started, retrieve the data with `./getdata.sh`.

### Training the model

`python -u main.py --epochs 14 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --data data/enwik8/ --save ENWIK8.pt --log-interval 10 --nhid 2048 --seed 5512 --optimizer lamb --bptt 1024 --warmup 800 --lr 2e-3 --emsize 1024 --nhid 4096 --nlayers 4 --batch_size 8`

The validation bpc should fall 1.332, 1.250, 1.212, 1.190, 1.175, 1.162, 1.154, 1.147, 1.140, 1.135, 1.131, 1.126, 1.123, 1.123 over the 14 epochs.

Each epoch took approximately 4400 seconds each for a total of 17.1 hours on a Titan V.

We then run one final training run:

`python -u main.py --epochs 14 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --data data/enwik8/ --save ENWIK8.pt --log-interval 10 --nhid 2048 --seed 5512 --optimizer lamb --bptt 1024 --warmup 800 --lr 2e-3 --emsize 1024 --nhid 4096 --nlayers 4 --batch_size 8 --resume ENWIK8.pt --lr 1e-3`

By dropping the learning rate from 2e-3 to 1e-3 we get validation bpcs falling 1.106, 1.104, 1.101, 1.101

The final test bpc should be 1.07.
