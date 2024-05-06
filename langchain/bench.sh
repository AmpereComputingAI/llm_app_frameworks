export AIO_PROCESS_MODE=1
export AIO_NUM_THREADS=32

mkdir -p data
echo "Small Benchark (AIO)"
rm ./data/*
echo "One file (small)"
cp ../data/state_of_the_union.txt ./data
ls -la ./data/
python run_benchmark.py small
echo "Two files (small)"
cp ../data/paul_graham_essay.txt ./data/
ls -la ./data/
python run_benchmark.py small
echo "Three files (small)"
cp ../data/pg1399.txt ./data/
ls -la ./data/
python run_benchmark.py small

echo "Base Benchmark (AIO)"
rm ./data/*
echo "One file (base)"
cp ../data/state_of_the_union.txt ./data
ls -la ./data/
python run_benchmark.py base
echo "Two files (base)"
cp ../data/paul_graham_essay.txt ./data/
ls -la ./data/
python run_benchmark.py base
echo "Three files (base)"
cp ../data/pg1399.txt ./data/
ls -la ./data/
python run_benchmark.py base
