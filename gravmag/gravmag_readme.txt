*********球棱柱tesseroid使用numba加速
_tesseroid_numba.py
tesseroid.py
	若只做正演，不计算核矩阵
	可以使用代码tesseroidforward.py


********直立棱柱prism使用Cython加速
_prism.pyx
prism.py
setup.py
	-------编译_prism.pyx文件--------
	需要提前安装Visual Studio,cython,mingw。
	在windows系统和linux系统解开对应的注释，之后
	在终端运行：

			python setup.py build_ext --inplace
	
	从而生成_
	prism.cp37-win_amd64.pyd文件（windows系统）
	或者
	_prism.cpython-37m-x86_64-linux-gnu.so文件（linux系统）
