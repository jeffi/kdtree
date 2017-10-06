#pragma once

#include <atomic>
#include <iostream>
#include <sstream>
#include <cassert>
#include <vector>

namespace test {

static std::atomic_uintmax_t g_assertionCount(0);

template <typename _T>
struct PrintWrap {
    const _T& value_;

    PrintWrap(const _T& v) : value_(v) {}
    
    template <typename _Char, typename _Traits>
    inline friend std::basic_ostream<_Char, _Traits>&
    operator << (std::basic_ostream<_Char, _Traits>& os, const PrintWrap& w) {
        return os << w.value_;
    }
};

template <>
struct PrintWrap<std::nullptr_t> {
    PrintWrap(std::nullptr_t v) {}
    
    template <typename _Char, typename _Traits>
    inline friend std::basic_ostream<_Char, _Traits>&
    operator << (std::basic_ostream<_Char, _Traits>& os, const PrintWrap& w) {
        return os << "nullptr";
    }    
};

template <>
struct PrintWrap<bool> {
    bool value_;
    PrintWrap(bool v) : value_(v) {}
    template <typename _Char, typename _Traits>
    inline friend std::basic_ostream<_Char, _Traits>&
    operator << (std::basic_ostream<_Char, _Traits>& os, const PrintWrap& w) {
        // could also use std::boolalpha
        return os << (w.value_ ? "true" : "false");
    }    
};


template <typename _T>
PrintWrap<_T> printWrap(const _T& t) { return PrintWrap<_T>(t); }

template <typename _T>
struct Expectation {
    _T value_;
    
    const char *expr_;
    const char *file_;
    int line_;

    mutable bool checked_;

    Expectation(const _T& value, const char *expr, const char *file, int line)
        : value_(value), expr_(expr), file_(file), line_(line), checked_(false)
    {
    }

    Expectation(_T&& value, const char *expr, const char *file, int line)
        : value_(std::move(value)), expr_(expr), file_(file), line_(line), checked_(false)
    {
    }
    
    ~Expectation() { assert(checked_); }

    template <typename _E>
    void fail(const char *op, const _E& expect) const {
        std::ostringstream str; 
        str << "Expected " << expr_ << op << printWrap(expect)
            << ", got " << printWrap(value_) << " at " << file_ << ':' << line_;
        throw std::runtime_error(str.str());
    }

#define DEFINE_OP(_op_)                                 \
    template <typename _E>                              \
    void operator _op_ (const _E& expect) const {       \
        assert(!checked_);                              \
        checked_ = true;                                \
        ++g_assertionCount;                             \
        if (!(value_ _op_ expect))                      \
            fail(" " #_op_ " ", expect);                \
    }

    DEFINE_OP(==)
    DEFINE_OP(!=)
    DEFINE_OP(<)
    DEFINE_OP(>)
    DEFINE_OP(<=)
    DEFINE_OP(>=)
#undef DEFINE_OP
};

class TestCase;

std::vector<TestCase*> g_testCases;

class TestCase {
    std::string name_;
    
public:
    TestCase(const std::string& name) : name_(name) {
        g_testCases.push_back(this);
    }

    virtual void testImpl() = 0;

    bool run() {
        auto assertionsBefore = g_assertionCount.load();
        try {
            auto start = std::chrono::high_resolution_clock::now();
            testImpl();
            auto elapsed = std::chrono::high_resolution_clock::now() - start;
            auto nAsserts = g_assertionCount.load() - assertionsBefore;
            std::ostringstream msg;
            msg.imbue(std::locale(""));
            msg << name_ << " \33[32mpassed\33[0m ("
                << nAsserts << " assertion" << (nAsserts == 1 ? "" : "s")
                << ", " << std::chrono::duration<double, std::micro>(elapsed).count()
                << " µs)\n";
            std::cout << msg.str() << std::flush;
            return true;
        } catch (const std::runtime_error& e) {
            std::ostringstream msg;
            msg.imbue(std::locale(""));
            msg << name_ << " \33[31;1mfailed.\33[0m\n\t" << e.what() << "\n";
            std::cout << msg.str() << std::flush;
            return false;
        }
    }
};

} // namespace test

#define EXPECT(expr) (::test::Expectation<typename std::decay<decltype(expr)>::type>(expr, #expr, __FILE__, __LINE__))

#define TEST_CASE(name)                                                 \
    struct test_case_ ## name : public ::test::TestCase {               \
        test_case_ ## name () : TestCase(#name) {}                      \
        void testImpl();                                                \
    };                                                                  \
    test_case_ ## name test_instance_ ## name;                          \
    void test_case_ ## name :: testImpl()


// template <typename _Fn>
// bool runTest(const std::string& name, _Fn fn) {
//     auto assertionsBefore = g_assertionCount.load();
//     try {
//         fn();
//         auto nAsserts = g_assertionCount.load() - assertionsBefore;
//         std::cout << name << " \33[32mpassed\33[0m ("
//                   << nAsserts << " assertion" << (nAsserts == 1 ? "" : "s")
//                   << ")" << std::endl;
//         return true;
//     } catch (const std::runtime_error& e) {
//         std::cerr << name << " \33[31;1mfailed.\33[0m\n\t" << e.what() << std::endl;
//         return false;
//     }
// }

int main(int argc, char* argv[]) {
    bool success = true;
    for (test::TestCase *test : test::g_testCases) {
        success &= test->run();
    }
    return success ? 0 : 1;
}

