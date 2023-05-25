C=g++
CFLAGS+=`pkg-config --cflags opencv` -std=c++11 -Wextra -pedantic -Wall -Werror
LDFLAGS+=`pkg-config --libs opencv`
PROG=prog
SRC=dlt_1
OBJS=$(SRC).o
.PHONY: clean
$(PROG): $(OBJS)
	$(CC) -o $(PROG) $(OBJS) $(LDFLAGS) -lstdc++
%.o: %.cpp
	$(CC) -c $(CFLAGS) $<
clean:
	rm -f $(OBJS) $(PROG) *.out
