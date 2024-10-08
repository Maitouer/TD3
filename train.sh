# python -m debugpy --listen 56777 --wait-for-client src/main.py \
python src/main.py -m \
    base.gpu_id=1 \
    data.dataset_name=ml-100k \
    train.epochs=30 \
    train.window=40 \
    train.batch_size=4096 \
    train.seq_lr=0.01 \
    train.net_lr=0.005 \
    SASRec.n_heads=1 \
    SASRec.n_layers=1 \
    SASRec.hidden_size=64 \
    SASRec.inner_size=128 \
    SASRec.attn_dropout_prob=0.5 \
    SASRec.hidden_dropout_prob=0.2 \
    learner_train.batch_size=25 \
    distilled_data.seq_num=50 \
    distilled_data.seq_len=50 \
    distilled_data.seq_dim1=96 \
    distilled_data.seq_dim2=96
