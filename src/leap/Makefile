OS := $(shell uname)
ARCH := $(shell uname -m)

ifeq ($(OS), Linux)
  ifeq ($(ARCH), x86_64)
    LEAP_LIBRARY := ./libLeap.so -Wl,-rpath,./
  else
    LEAP_LIBRARY := ./libLeap.so -Wl,-rpath,./
  endif
else
  # OS X
  LEAP_LIBRARY := ./libLeap.dylib
endif

Sample: Read.cpp
	$(CXX) -Wall -g -I./ Read.cpp -o Read $(LEAP_LIBRARY)
ifeq ($(OS), Darwin)
	install_name_tool -change @loader_path/libLeap.dylib ./libLeap.dylib Read
endif

clean:
	rm -rf Read Read.dSYM
