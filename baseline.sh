# python -m debugpy --listen 56777 --wait-for-client src/baseline.py \
python src/baseline.py -m \
    base.gpu_id=0 \
    data.dataset_name=ml-100k \
