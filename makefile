CC = clang++
LFLAGS = -std=c++11 -Wall

all: LFLAGS += -Ofast -march=native
all: lstm-compress

debug: LFLAGS += -ggdb
debug: lstm-compress

lstm-compress: src/coder/decoder.cpp src/coder/decoder.h src/coder/encoder.cpp src/coder/encoder.h src/lstm/byte-model.cpp src/lstm/byte-model.h src/lstm/lstm.cpp src/lstm/lstm.h src/lstm/lstm-layer.cpp src/lstm/lstm-layer.h src/lstm/sigmoid.cpp src/lstm/sigmoid.h src/predictor.cpp src/predictor.h src/preprocess/dictionary.cpp src/preprocess/dictionary.h src/preprocess/preprocessor.cpp src/preprocess/preprocessor.h src/runner.cpp
	$(CC) $(LFLAGS) src/coder/decoder.cpp src/coder/encoder.cpp src/lstm/byte-model.cpp src/lstm/lstm.cpp src/lstm/lstm-layer.cpp src/lstm/sigmoid.cpp src/predictor.cpp src/preprocess/dictionary.cpp src/preprocess/preprocessor.cpp src/runner.cpp -o lstm-compress

clean:
	rm -f lstm-compress
