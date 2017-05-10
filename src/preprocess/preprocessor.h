#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <stdio.h>

namespace preprocessor {

void encode(FILE* in, FILE* out, int n, std::string temp_path,
    FILE* dictionary);

void decode(FILE* in, FILE* out, std::string temp_path, FILE* dictionary);

}

#endif
