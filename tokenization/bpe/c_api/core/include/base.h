// base.h
#ifndef BASE_H
#define BASE_H

#include <vector>
#include <map>
#include <unordered_map>
#include <stdlib.h>
#include <cstdint>
#include <algorithm>

using namespace std;

/*
the hash function for pair in unordered_map
*/
struct pair_hash {
    template <class T1, class T2>
    size_t operator()(const pair<T1, T2>& pair) const {
        auto h1 = hash<T1>{}(pair.first);
        auto h2 = hash<T2>{}(pair.second);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

void merge_bytes_pair(
    vector<int> &byte_list,
    const pair<int, int>& pair_,
    const int token_id
);

void get_bytes_pair_counts(
    const vector<int>& byte_list,
    unordered_map<pair<int, int>, int, pair_hash> &pair_counts
);

void get_top_pair(
    const unordered_map<pair<int, int>, int, pair_hash>& pair_counts,
    pair<int, int> &top_pair
);

// void get_bytes_pair_counts(
//     vector<int> &byte_list,
//     map<pair<int, int>, int> &pair_counts
// );

// void get_top_pair(
//     map<pair<int, int>, int> &pair_counts,
//     pair<int, int> &top_pair
// );

#endif