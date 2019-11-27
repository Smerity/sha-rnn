# Single Headed Attention RNN

For full details see the paper [Single Headed Attention RNN: Stop Thinking With Your Head](https://arxiv.org/abs/1911.11423).

In summary, "stop thinking with your (attention) head".

- Obtain strong results on a byte level language modeling dataset (enwik8) in under 24 hours on a single GPU (12GB Titan V)
- Support long range dependencies (up to 5000 tokens) without increasing compute time or memory usage substantially by using a simpler attention mechanism
- Avoid the fragile training process required by standard Transformer models such as a long warmup
- Back off toward a standard LSTM allowing you to drop retained memory states (needed for a Transformer model) if memory becomes a major constraint
- Provide a smaller model that features only standard components such as the LSTM, single headed attention, and feed-forward modules such that they can easily be productionized using existing optimized tools and exported to various formats (i.e. ONNX)

| Model                             | Test BPC | Params | LSTM Based |
|-----------------------------------|----------|--------|------------|
| Krause mLSTM                      | 1.24     | 46M    | ✔          |
| AWD-LSTM                          | 1.23    | 44M    | ✔          |
| **SHA-LSTM**                          | 1.07     | 63M    | ✔          |
| 12L Transformer-XL                | 1.06     | 41M    |            |
| 18L Transformer-XL                | 1.03     | 88M    |            |
| Adaptive Span Transformer (Small) | 1.02     | 38M    |            |

Whilst the model is still quite some way away from state of the art (~0.98 bpc) the model is low resource and high efficiency without having yet been optimized to be so.
The model was trained in under 24 hours on a single GPU with the [Adaptive Span Transformer](https://github.com/facebookresearch/adaptive-span) (small) being the only recent Transformer model to achieve similar levels of training efficiency.

## To recreate

### Setup

To get started:

- Retrieve the data with `./getdata.sh`
- Install PyTorch version 1.2
- Install Nvidia's [AMP](https://github.com/NVIDIA/apex)
- Install the minimum trust variant of LAMB from [Smerity's PyTorch-LAMB](https://github.com/Smerity/pytorch-lamb)

### Training the model

By default the model trains the minimal single headed attention model from the paper, inserting a lone attention mechanism in the second last layer of a four layer LSTM.
Sadly there are no command line options for running the other models.
The code is not kind.
I'll be performing a re-write in the near future meant for long term academic and industrial use - contact me if you're interested :)

`python -u main.py --epochs 14 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --data data/enwik8/ --save ENWIK8.pt --log-interval 10 --seed 5512 --optimizer lamb --bptt 1024 --warmup 800 --lr 2e-3 --emsize 1024 --nhid 4096 --nlayers 4 --batch_size 16`

When the training slows down it is suggested to run a second pass with a halved learning rate:

`python -u main.py --epochs 14 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --data data/enwik8/ --save ENWIK8.pt --log-interval 10 --seed 5512 --optimizer lamb --bptt 1024 --warmup 800 --lr 2e-3 --emsize 1024 --nhid 4096 --nlayers 4 --batch_size 8 --resume ENWIK8.pt --lr 1e-3 --seed 125`

The final test bpc should be approximately 1.07.
