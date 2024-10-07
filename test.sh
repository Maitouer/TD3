# python -m debugpy --listen 56766 --wait-for-client src/main.py -m \
python src/main.py \
    train.skip_train=True \
    model.use_pretrained_model=false \
    model.use_pretrained_embed=True \
    model.freeze_pretrained_embed=True \
    distilled_data.pretrained_data_path=./save/SASRec.ml-100k/num_50.len_50.dim_96.seq_lr_0.01.net_lr_0.005.epochs_30/2024-10-06.19-01-19/checkpoints/best-ckpt \
    base.gpu_id=0 \
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
    learner_train.train_step=600 \
    distilled_data.seq_num=50 \
    distilled_data.seq_len=50 \
    distilled_data.seq_dim1=96 \
    distilled_data.seq_dim2=96
