cd external

# eigen
wget http://bitbucket.org/eigen/eigen/get/3.1.1.tar.bz2
tar xjvf 3.1.1.tar.bz2
mkdir eigen
mv eigen-*/Eigen eigen
rm 3.1.1.tar.bz2
rm -rf eigen-*

# liblbfgs
wget https://github.com/downloads/chokkan/liblbfgs/liblbfgs-1.10.tar.gz
tar xzfv liblbfgs-1.10.tar.gz
mv liblbfgs-1.10 liblbfgs
rm liblbfgs-1.10.tar.gz

# gtest
wget http://googletest.googlecode.com/files/gtest-1.6.0.zip
unzip gtest-1.6.0.zip
mkdir gtest
mv gtest-1.6.0/src gtest
mv gtest-1.6.0/include gtest
rm -rf gtest-1.6.0/
rm gtest-1.6.0.zip




cd ..
