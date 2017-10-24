TARGET = ../build
testsrc := $(wildcard *_test.cpp)
testexe := $(patsubst %.cpp,$(TARGET)/%,$(testsrc))

# CXX=/opt/local/bin/clang++-mp-5.0
CXXFLAGS += -std=c++14 -I../src -I/opt/local/include/eigen3 -Wall -pedantic -O3 -g

all: run_tests

# remove default rule
%: %.cpp

# Compile directly to executable (no intermediate .o files)
$(TARGET)/%: %.cpp $(TARGET)/%.d
	@echo "compiling" $@
	@mkdir -p $(@D)
	@$(CXX) -MMD -MF $(patsubst %.cpp,$(TARGET)/%.dtmp,$<) $(CXXFLAGS) -o $@ $<
	@mv $(patsubst %.cpp,$(TARGET)/%.dtmp,$<) $(patsubst %.cpp,$(TARGET)/%.d,$<)
	@touch $@

# Run each test.  If successful, touch the .success file to mark its success
$(TARGET)/%.success: $(TARGET)/%
	@echo "TESTS " $(patsubst $(TARGET)/%.success,%,$@)
	@set -o pipefail ; $< | tee $<.log && touch $@

%.d: ;
.PRECIOUS: %.d

.PHONY: all run_tests compile_tests clean

run_tests: $(patsubst %.cpp,$(TARGET)/%.success,$(testsrc))

compile_tests: $(testexe)

clean:
	$(RM) -r $(TARGET)

# Include generated dependencies
include $(patsubst %.cpp,$(TARGET)/%.d,$(testsrc))
