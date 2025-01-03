# makefile
CXX = g++
CXXFLAGS = -Icore/include -std=c++17 -O3
SRC = core/src/base.cpp core/src/basic.cpp tests/test.cpp
TARGET = test_bpe

# make
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)

test: $(TARGET)
	@echo "Running tests..."
	@./$(TARGET)