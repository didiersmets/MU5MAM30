# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Iinclude -std=c++17

# Directories
SRC_DIR = src
INCLUDE_DIR = include
RUN_DIR = run
BUILD_DIR = build

# Source files
SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp)

# Object files
OBJ_FILES = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRC_FILES))

# Executable files
RUN_FILES = $(wildcard $(RUN_DIR)/*.cpp)
EXECUTABLES = $(patsubst $(RUN_DIR)/%.cpp, $(BUILD_DIR)/%, $(RUN_FILES))

# Default target
all: $(EXECUTABLES)

# Rule to build executables
$(BUILD_DIR)/%: $(RUN_DIR)/%.cpp $(OBJ_FILES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< $(OBJ_FILES) -o $@

# Rule to build object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean up build directory
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
# Rule to build a single executable
build/%: $(RUN_DIR)/%.cpp $(OBJ_FILES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< $(OBJ_FILES) -o $@

# Rule to build a single object file
build/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Help target
help:
	@echo "Usage:"
	@echo "  make            - Build all executables"
	@echo "  make build/FILE - Build a single executable (e.g., make build/myprogram)"
	@echo "  make clean      - Remove all build files"