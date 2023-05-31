CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11
LIBS = `pkg-config --cflags --libs opencv`

SRC = RANSAC_own.cpp

TARGET = RANSAC_own

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGET)

