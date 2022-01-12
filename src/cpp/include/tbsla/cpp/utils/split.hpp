#ifndef SPLIT_HPP
#define SPLIT_HPP
#include <string>
#include <vector>

namespace tbsla { namespace utils { namespace io {

  std::vector<std::string> split(const std::string& str, const std::string& delim); 

}}}
#endif