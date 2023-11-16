# Makefile for the C++ example

# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -g

# Build target executable
TARGET = myProgram

# Source files
SOURCES = main.cpp myClass.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Default target
all: $(TARGET)

# Linking
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

# Compiling
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

# Clean up
clean:
	rm -f $(OBJECTS) $(TARGET)

