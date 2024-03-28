echo "Small Benchark (OS)"
rm ./data/*
echo "One file (small)"
cp state_of_the_union.txt ./data
ls -la ./data/
python os_small.py
echo "Two files (small)"
cp paul_graham_essay.txt ./data/
ls -la ./data/
python os_small.py
echo "Three files (small)"
cp pg1399.txt ./data/
ls -la ./data/
python os_small.py

echo "Base Benchmark (OS)"
rm ./data/*
echo "One file (base)"
cp state_of_the_union.txt ./data
ls -la ./data/
python os_base.py
echo "Two files (base)"
cp paul_graham_essay.txt ./data/
ls -la ./data/
python os_base.py
echo "Three files (base)"
cp pg1399.txt ./data/
ls -la ./data/
python os_base.py

