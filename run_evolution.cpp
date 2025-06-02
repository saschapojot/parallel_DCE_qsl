#include "./evolution/evolution.hpp"

int main(int argc, char *argv[])
{

    if (argc != 2) {
        std::cout << "wrong arguments" << std::endl;
        std::exit(2);
    }
    auto evo_obj=evolution(std::string(argv[1]));
    evo_obj.init();
    evo_obj.run_and_save_H1R_only();
}