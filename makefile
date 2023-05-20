CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11
LIBS = `pkg-config --cflags --libs opencv`

SRC = main.cpp

TARGET = MarkerPoseEstimation

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGET)

