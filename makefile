CC = g++
CFLAGS = -std=c++11 -Wall -c
LFLAGS = -std=c++11 -Wall

OBJS = build/lstm.o build/layer.o

all: CFLAGS += -Ofast -march=native -s
all: LFLAGS += -Ofast -march=native -s
all: build runner

debug: CFLAGS += -ggdb
debug: LFLAGS += -ggdb
debug: build runner

runner: $(OBJS) src/runner.cpp
	$(CC) $(LFLAGS) $(OBJS) src/runner.cpp -o runner

build/lstm.o: src/lstm.h src/lstm.cpp src/layer.h
	$(CC) $(CFLAGS) src/lstm.cpp -o build/lstm.o

build/layer.o: src/layer.h src/layer.cpp
	$(CC) $(CFLAGS) src/layer.cpp -o build/layer.o

build:
	mkdir -p build/

clean:
	rm -f -r build/* runner
