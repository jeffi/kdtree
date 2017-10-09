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

// helper to print elements of a tuple
template <int _index, typename ... _Types>
struct PrintTupleElements {
    typedef std::tuple<_Types...> Tuple;
    
    template <typename _Char, typename _Traits>
    static void apply(std::basic_ostream<_Char, _Traits>& os, const Tuple& t) {
        if (_index > 0) os << ", ";
        os << PrintWrap<typename std::tuple_element<_index, Tuple>::type>(std::get<_index>(t));
        PrintTupleElements<_index+1, _Types...>::apply(os, t);
    }
};

// helper to print elements of a tuple, base case.
template <typename ... _Types>
struct PrintTupleElements<sizeof...(_Types), _Types...> {
    typedef std::tuple<_Types...> Tuple;
    
    template <typename _Char, typename _Traits>
    static void apply(std::basic_ostream<_Char, _Traits>& os, const Tuple& t) {}
};

// tuples get printed as [<0>, <1>, ...], where <n> is the element at
// index <n>.
template <typename ... _Types>
struct PrintWrap<std::tuple<_Types...>> {
    typedef std::tuple<_Types...> Tuple;
    
    const Tuple& value_;
    PrintWrap(const Tuple& v) : value_(v) {}
    
    template <typename _Char, typename _Traits>
    inline friend std::basic_ostream<_Char, _Traits>&
    operator << (std::basic_ostream<_Char, _Traits>& os, const PrintWrap& w) {
        os << "[";
        PrintTupleElements<0, _Types...>::apply(os, w.value_);
        return os << "]";
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

    const std::string& name() const {
        return name_;
    }

    virtual void testImpl() = 0;

    template <typename _Reason>
    static void failed(const std::string& name, const _Reason& reason) {
        std::ostringstream msg;
        msg.imbue(std::locale(""));
        msg << name << " \33[31;1mfailed ⚠\33[0m\n\t" << reason << "\n";
        std::cout << msg.str() << std::flush;
    }
    
    bool run() {
        auto assertionsBefore = g_assertionCount.load();
        try {
            auto start = std::chrono::high_resolution_clock::now();
            testImpl();
            auto elapsed = std::chrono::high_resolution_clock::now() - start;
            auto nAsserts = g_assertionCount.load() - assertionsBefore;
            std::ostringstream msg;
            msg.imbue(std::locale(""));
            msg << name_ << " \33[32mpassed ✓\33[0m ("
                << (nAsserts ? "" : "\33[31m")
                << nAsserts << " assertion" << (nAsserts == 1 ? "" : "s")
                << (nAsserts ? "" : "\33[0m")
                << ", " << std::chrono::duration<double, std::milli>(elapsed).count()
                << " ms)\n";
            std::cout << msg.str() << std::flush;
            return true;
        } catch (const std::runtime_error& e) {
            failed(name_, e.what());
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
    using namespace test;
    
    int testCases;
    int passed = 0;
    if (argc > 1) {
        testCases = argc - 1;
        for (int i=1 ; i<argc ; ++i) {
            std::string name = argv[i];
            
            auto it = std::find_if(
                g_testCases.begin(), g_testCases.end(), [&] (auto t) { return t->name() == name; });
            
            if (it == g_testCases.end()) {
                TestCase::failed(name, "no test found with matching name");
            } else {
                passed += (*it)->run();
            }
        }
    } else {
        testCases = test::g_testCases.size();
        for (test::TestCase *test : test::g_testCases) {
            passed += test->run();
        }
    }
    
    std::cout << passed << " of "
              << testCases << " test" << (testCases == 1?"":"s") << " passed."
              << std::endl;
    return testCases == passed ? 0 : 1;
}

