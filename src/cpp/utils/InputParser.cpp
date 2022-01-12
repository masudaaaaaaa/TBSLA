#include <tbsla/cpp/utils/InputParser.hpp>
#include <algorithm>

InputParser::InputParser(int &argc, char **argv) {
  this->params.reserve(argc - 1);
  for (int i=1; i < argc; ++i)
    this->params.push_back(std::string(argv[i]));
}

std::string InputParser::get_opt(const std::string &option, const std::string default_opt) const {
  std::vector<std::string>::const_iterator itr;
  itr = std::find(this->params.begin(), this->params.end(), option);
  if (itr != this->params.end() && ++itr != this->params.end()){
      return *itr;
  }
  return default_opt;
}

bool InputParser::has_opt(const std::string &option) const {
  return std::find(this->params.begin(), this->params.end(), option) != this->params.end();
}
