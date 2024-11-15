# Directory where the source files are located
SRC_DIR = src

# Directory of executables
BUILD_DIR = build

# Directory where the include files are located
INC_DIR = include

# Directory to use for other include files of the professor
PROF_INC_DIR = ../../include/common

# Compiler and compiler flags
CXX = g++
CXXFLAGS = -Wall -g -I$(INC_DIR) -I$(PROF_INC_DIR)

# Default target
.PHONY: all
all: help

# Rule for compiling a single .cpp file from the src directory
# The $@ is the target (e.g., tp1 or test), and $< is the first dependency (the .cpp file)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule for linking object files and generating the executable
# This is for all .o files corresponding to .cpp files in the src directory
$(BUILD_DIR)/%: $(BUILD_DIR)/%.o
	$(CXX) $(CXXFLAGS) $< -o $@

# Custom target to compile a specific .cpp file
.PHONY: compile
compile: $(SRC_DIR)/$(FILE).cpp
	$(MAKE) $(BUILD_DIR)/$(FILE)

# Help target to display usage information
.PHONY: help
help:
	@echo "Usage: make FILE=<filename> compile"
	@echo "Example: make FILE=tp1 compile"

# Clean up build artifacts
.PHONY: clean
clean:
	rm -f $(BUILD_DIR)/* 