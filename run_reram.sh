mkdir -p logs
mkdir -p comp_reram/logs

if [[ "$IMAGENET" == "" ]]; then
	echo "Please set IMAGENET environment variable pointing to imagenet dataset."
	echo "Expected: \$IMAGENET/val/<classes>"
	exit 1
fi

python3 -u -W ignore reram_search.py     \
 --dataset imagenet                 \
 --n_worker 32                      \
 --data_bsize 256                   \
 --train_size 20000                 \
 --val_size 10000                   \
 --gpu_id 1                         \
 --batch-size 1

