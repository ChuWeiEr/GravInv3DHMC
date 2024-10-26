#!/usr/bin/python  
#python version: 3.7  
#Filename: Setup_prism.py  
   
# Run as:    
#    python setup.py build_ext --inplace    
     
import sys    
import numpy as np
sys.path.insert(0, "..")    
     
from distutils.core import setup    
from distutils.extension import Extension    
from Cython.Build import cythonize    
from Cython.Distutils import build_ext  


# windows
#ext_module = cythonize("_prism.pyx")    
ext_module = Extension(  
                        "_prism",  
            ["_prism.pyx"],  
            extra_compile_args=["/openmp"],  
            extra_link_args=["/openmp"],  
            )  
            
setup(  
    cmdclass = {'build_ext': build_ext},  
        ext_modules = [ext_module],   
        include_dirs=[np.get_include()]
)  

''' 
#linux
ext_module = Extension(  
                        "_prism",  
            ["_prism.pyx"])  
            
setup(  
    cmdclass = {'build_ext': build_ext},  
        ext_modules = [ext_module],   
        include_dirs=[np.get_include()]
)  
'''
