#ifndef TBSLA_CPP_Reduction
#define TBSLA_CPP_Reduction

namespace tbsla { namespace cpp { namespace reduction {

template <class T>
struct array {
  public:
    array(T* _v, std::size_t _len) : v(_v), len(_len) { del = false;}
    array(std::size_t _len) : len(_len) {
      v = new T[len]();
      del = true;
    }
    ~array() {
      if (del and v) {
        delete[] v;
      }
    }
    void add(tbsla::cpp::reduction::array<T> &vin) {
      #pragma omp parallel for
      for(int i = 0; i < this->len; i++) {
        this->v[i] += vin[i];
      }
    }
    std::size_t size() {return len;}
    T& operator[](int index) {
      return v[index];
    }

  private:
    T* v;
    std::size_t len;
    bool del;
};


}}}

#endif
