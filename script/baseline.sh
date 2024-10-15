python src/baseline.py \
    base.gpu_id=1 \
    data.dataset_name=epinions \
    SASRec.n_heads=1 \
    SASRec.n_layers=2 \
    SASRec.hidden_size=32 \
    SASRec.inner_size=32 \
    SASRec.attn_dropout_prob=0.5 \
    SASRec.hidden_dropout_prob=0.2
