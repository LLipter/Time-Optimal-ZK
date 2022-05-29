echo "thread=1"
RAYON_NUM_THREADS=1 cargo run > thread-1.txt
echo "thread=8"
RAYON_NUM_THREADS=8 cargo run > thread-8.txt