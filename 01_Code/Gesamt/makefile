C=g++
CFLAGS+=`pkg-config --cflags opencv` -std=c++11 -pedantic -Wextra -Wall -Werror
LDFLAGS+=`pkg-config --libs opencv`
PROG=prog
SRC=markerPoseEstimation
OBJS=$(SRC).o
.PHONY: clean
$(PROG): $(OBJS)
	$(CC) -o $(PROG) $(OBJS) $(LDFLAGS) -lstdc++ -lm
%.o: %.cpp
	$(CC) -c $(CFLAGS) $<
clean:
	rm -f $(OBJS) $(PROG) *.out
