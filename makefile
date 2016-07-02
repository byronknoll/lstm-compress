CC = g++
CFLAGS = -std=c++11 -Wall -c
LFLAGS = -std=c++11 -Wall

OBJS = build/lstm-compress.o

all: CFLAGS += -Ofast -march=native -s
all: LFLAGS += -Ofast -march=native -s
all: build runner

debug: CFLAGS += -ggdb
debug: LFLAGS += -ggdb
debug: build runner generator

runner: $(OBJS) src/runner.cpp
	$(CC) $(LFLAGS) $(OBJS) src/runner.cpp -o runner

build/lstm-compress.o: src/lstm-compress.h src/lstm-compress.cpp
	$(CC) $(CFLAGS) src/lstm-compress.cpp -o build/lstm-compress.o

build:
	mkdir -p build/

clean:
	rm -f -r build/* runner
