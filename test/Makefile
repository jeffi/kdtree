TARGET = build
testsrc := $(wildcard *_test.cpp)
testexe := $(patsubst %.cpp,$(TARGET)/%,$(testsrc))

CXXFLAGS += -std=c++14 -I../src -Wall -pedantic -O3 -g
PKG_CONFIG ?= pkg-config

# Check if pkg-config is available, if so, use it to check for dependencies
ifneq ($(shell command -v $(PKG_CONFIG)),)

ifeq ($(shell $(PKG_CONFIG) --exists eigen3 && echo 1),1)
CXXFLAGS += `pkg-config --cflags eigen3`
else
MISSING += eigen3
endif

# else if pkg-config is not available, test for dependencies directly
else # no pkg_config, manually test

ifeq ($(shell echo $$'\#include <Eigen/Dense>\nint main(){}' | $(CXX) $(CXXFLAGS) -c -o/dev/null -xc++ - 2>/dev/null || echo 0),0)
MISSING += eigen3
endif

endif
#end of dependencies checking

# check results of dependency checks
ifeq ($(MISSING),)
all: run_tests
else
all:
	@echo "MISSING DEPENDENCIES"
	@echo
	@echo "Your system is missing the following dependencies, please install the following"
	@echo "dependencies and make again"
	@echo
	@echo "  $(MISSING)"
endif


# remove default rule
%: %.cpp

# Compile directly to executable (no intermediate .o files)
$(TARGET)/%: %.cpp $(TARGET)/%.d
	@echo "Compiling" $@
	@mkdir -p $(@D)
	@$(CXX) -MMD -MF $(patsubst %.cpp,$(TARGET)/%.dtmp,$<) $(CXXFLAGS) -o $@ $<
	@mv $(patsubst %.cpp,$(TARGET)/%.dtmp,$<) $(patsubst %.cpp,$(TARGET)/%.d,$<)
	@touch $@

# Run each test.  If successful, touch the .success file to mark its success
$(TARGET)/%.success: $(TARGET)/%
	@echo "RUNNING TEST " $(patsubst $(TARGET)/%.success,%,$@)
	@set -o pipefail ; $< | tee $<.log && touch $@

%.d: ;
.PRECIOUS: %.d

.PHONY: all run_tests compile_tests clean

run_tests: $(patsubst %.cpp,$(TARGET)/%.success,$(testsrc))

compile_tests: $(testexe)

clean:
	$(RM) -r $(TARGET)

# Include generated dependencies
-include $(patsubst %.cpp,$(TARGET)/%.d,$(testsrc))
