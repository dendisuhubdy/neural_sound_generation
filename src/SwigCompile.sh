swig -c++ -python -o LeapPython.cpp -interface LeapPython Leap.i
c++ -arch x86_64 -I/usr/local/Cellar/python/3.7.1/Frameworks/Python.framework/Versions/3.7/include/python3.7m LeapPython.cpp libLeap.dylib /usr/local/Cellar/python/3.7.1/Frameworks/Python.framework/Versions/3.7/lib/libpython3.7.dylib -shared -o LeapPython.so
