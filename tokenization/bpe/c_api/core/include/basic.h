// basic.h
#ifndef BASIC_H
#define BASIC_H

#include <vector>
#include <map>
#include <unordered_map>
#include <stdlib.h>
#include <cstdint>

using namespace std;

vector<vector<int>> bytes_pair_encoding(
    vector<vector<int>>& tokens,
    int vocab_size
);

#endif