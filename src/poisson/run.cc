//
// Created by Sigmund Eggen Holm on 17/10/2020.
//
#include <iostream>

#include "poisson.h"

using namespace dealii;


int main() {
    std::cout << "PoissonNitsche" << std::endl;
    {
        Poisson<2> poisson(1);
        poisson.run();
    }
}
