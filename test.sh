# python -m debugpy --listen 56766 --wait-for-client src/main.py -m \
python src/main.py -m \
    train.skip_train=True \
    model.use_pretrained_model=false \
    model.use_pretrained_embed=True \
    model.freeze_pretrained_embed=True \
    distilled_data.pretrained_data_path=./save/SASRec.epinions/num_50.len_20.dim_32.seq_lr_0.01.net_lr_0.003.epochs_30/2024-10-06.13-14-59/checkpoints/best-ckpt \
    base.gpu_id=0 \
    data.dataset_name=epinions \
    train.epochs=30 \
    train.window=40 \
    train.batch_size=2048 \
    train.seq_lr=0.01 \
    train.net_lr=0.003 \
    SASRec.n_heads=1 \
    SASRec.n_layers=2 \
    SASRec.hidden_size=32 \
    SASRec.inner_size=32 \
    SASRec.attn_dropout_prob=0.5 \
    SASRec.hidden_dropout_prob=0.2 \
    learner_train.batch_size=25 \
    distilled_data.seq_num=50 \
    distilled_data.seq_len=20 \
    distilled_data.seq_dim1=32 \
    distilled_data.seq_dim2=32