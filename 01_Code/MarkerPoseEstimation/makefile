CC=g++
CFLAGS+=`pkg-config --cflags opencv` -std=c++11 -pedantic -Wextra -Wall -Werror
LDFLAGS+=`pkg-config --libs opencv`
PROG=prog
SRC=markerPoseEstimation
.PHONY: clean
$(PROG): $(SRC).cpp
	$(CC) $(CFLAGS) -o $(PROG) $(SRC).cpp $(LDFLAGS) -lstdc++ -lm
clean:
	rm -f $(PROG) *.out
