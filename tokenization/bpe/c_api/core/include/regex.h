// regex.h
#ifndef REGEX_H
#define REGEX_H

#include <vector>
#include <map>
#include <stdlib.h>

using namespace std;

map<pair<int, int>, int> regex_bytes_pair_encoding(
    vector<int> &byte_list,
    int vocab_size
);

#endif