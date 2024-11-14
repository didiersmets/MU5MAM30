# Directory where the source files are located
SRC_DIR = src

# Directory where the include files are located
INC_DIR = include

# Directory to use for other include files of the professor
PROF_INC_DIR = ../include/common

# Compiler and compiler flags
CXX = g++
CXXFLAGS = -Wall -g -I$(INC_DIR) -I$(PROF_INC_DIR)

# Default target
.PHONY: all
all: $(SRC_DIR)/tp1.cpp 

# Rule for compiling a single .cpp file from the src directory
# The $@ is the target (e.g., tp1 or test), and $< is the first dependency (the .cpp file)
%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for linking object files and generating the executable
# This is for all .o files corresponding to .cpp files in the src directory
%: %.o
	$(CXX) $(CXXFLAGS) $< -o $@

# Run the program after building it
run-%: %
	./$<

# Clean up generated files
clean:
	rm -f $(SRC_DIR)/*.o tp1 test
