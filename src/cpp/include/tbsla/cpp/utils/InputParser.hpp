#include <string>
#include <vector>

class InputParser {
  public:
    InputParser(int &argc, char **argv);
    std::string get_opt(const std::string &option, const std::string default_opt = std::string("")) const;
    bool has_opt(const std::string &option) const;
  private:
    std::vector<std::string> params;
};
