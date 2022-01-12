#ifndef CPUSET_TO_CSTR_HPP
#define CPUSET_TO_CSTR_HPP
#include <sched.h>

namespace tbsla { namespace cpp { namespace utils {

  char *cpuset_to_cstr(cpu_set_t *mask, char *str);

}}}
#endif
