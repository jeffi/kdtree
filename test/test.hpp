#pragma once

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


#define EXPECT(expr) (Expectation<typename std::decay<decltype(expr)>::type>(expr, #expr, __FILE__, __LINE__))

template <typename _Fn>
bool runTest(const std::string& name, _Fn fn) {
    auto assertionsBefore = g_assertionCount.load();
    try {
        fn();
        auto nAsserts = g_assertionCount.load() - assertionsBefore;
        std::cout << name << " \33[32mpassed\33[0m ("
                  << nAsserts << " assertion" << (nAsserts == 1 ? "" : "s")
                  << ")" << std::endl;
        return true;
    } catch (const std::runtime_error& e) {
        std::cerr << name << " \33[31;1mfailed.\33[0m\n\t" << e.what() << std::endl;
        return false;
    }
}    
