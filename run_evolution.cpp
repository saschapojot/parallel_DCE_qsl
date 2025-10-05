#include "./evolution/evolution.hpp"

int main(int argc, char *argv[])
{

    if (argc != 2) {
        std::cout << "wrong arguments" << std::endl;
        std::exit(2);
    }
    auto evo_obj=evolution(std::string(argv[1]));

}