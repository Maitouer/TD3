# python -m debugpy --listen 56777 --wait-for-client src/baseline.py \
python src/baseline.py -m \
    base.gpu_id=0 \
    data.dataset_name=ml-100k \
    SASRec.n_heads=1 \
    SASRec.n_layers=1 \
    SASRec.hidden_size=32 \
    SASRec.inner_size=64 \
    SASRec.attn_dropout_prob=0.5 \
    SASRec.hidden_dropout_prob=0.2
