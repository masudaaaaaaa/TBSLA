#include <tbsla/cpp/utils/InputParser.hpp>
#include <iostream>

int main() {

  char * argv[] = {(char *)"app_name", (char *)"-f", (char *)"test"};
  int argc = 3;

  InputParser input(argc, argv);

  if(input.get_opt("-f") != "test") {
    return 1;
  }
  if(input.get_opt("-i", "default") != "default") {
    return 2;
  }
  if(input.has_opt("-g")) {
    return 3;
  }
  if(!input.has_opt("-f")) {
    return 4;
  }
}
