CC = g++
CFLAGS = -std=c++11 -Wall -c
LFLAGS = -std=c++11 -Wall

OBJS = build/preprocessor.o build/encoder.o build/decoder.o build/predictor.o build/sigmoid.o build/byte-model.o build/lstm.o build/lstm-layer.o

all: CFLAGS += -Ofast
all: LFLAGS += -Ofast
all: build lstm-compress

debug: CFLAGS += -ggdb
debug: LFLAGS += -ggdb
debug: build lstm-compress

lstm-compress: $(OBJS) src/runner.cpp
	$(CC) $(LFLAGS) $(OBJS) src/runner.cpp -o lstm-compress

build/preprocessor.o: src/preprocess/preprocessor.h src/preprocess/preprocessor.cpp src/preprocess/textfilter.cpp
	$(CC) $(CFLAGS) src/preprocess/preprocessor.cpp -o build/preprocessor.o

build/encoder.o: src/coder/encoder.h src/coder/encoder.cpp src/predictor.h
	$(CC) $(CFLAGS) src/coder/encoder.cpp -o build/encoder.o

build/decoder.o: src/coder/decoder.h src/coder/decoder.cpp src/predictor.h
	$(CC) $(CFLAGS) src/coder/decoder.cpp -o build/decoder.o

build/predictor.o: src/predictor.h src/predictor.cpp src/lstm/byte-model.h
	$(CC) $(CFLAGS) src/predictor.cpp -o build/predictor.o

build/byte-model.o: src/lstm/byte-model.h src/lstm/byte-model.cpp src/lstm/lstm.h
	$(CC) $(CFLAGS) src/lstm/byte-model.cpp -o build/byte-model.o

build/lstm.o: src/lstm/lstm.h src/lstm/lstm.cpp src/lstm/lstm-layer.h
	$(CC) $(CFLAGS) src/lstm/lstm.cpp -o build/lstm.o

build/lstm-layer.o: src/lstm/lstm-layer.h src/lstm/lstm-layer.cpp src/lstm/sigmoid.h
	$(CC) $(CFLAGS) src/lstm/lstm-layer.cpp -o build/lstm-layer.o

build/sigmoid.o: src/lstm/sigmoid.h src/lstm/sigmoid.cpp
	$(CC) $(CFLAGS) src/lstm/sigmoid.cpp -o build/sigmoid.o

build:
	mkdir -p build/

clean:
	rm -f -r build/* lstm-compress
