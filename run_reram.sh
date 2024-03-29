mkdir -p logs
mkdir -p comp_reram/logs

if [[ "$IMAGENET" == "" ]]; then
	echo "Please set IMAGENET environment variable pointing to imagenet dataset."
	echo "Expected: \$IMAGENET/val/<classes>"
	exit 1
fi

python3 -u -W ignore reram_search.py --batch-size 1
