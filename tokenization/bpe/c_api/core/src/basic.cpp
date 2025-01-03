// basic.cpp
#include "../include/basic.h"
#include "../include/base.h"

using namespace std;

/*
run BPE algorithm to build merge_map

@param tokens: list of tokens
@param vocab_size: vocabulary size
@return: merge_map
*/
vector<vector<int>> bytes_pair_encoding(
    vector<vector<int>>& tokens,
    int vocab_size
){
    // init merge_map
    vector<vector<int>> merge_map;
    merge_map.reserve(vocab_size);
    pair<int, int> top_pair;
    unordered_map<pair<int, int>, int, pair_hash> pair_counts;

    // number of merge rounds
    int num_merge_rounds = vocab_size - 256;
    int token_id = 256;

    // iterate over tokens
    for(int i = 0; i < num_merge_rounds; i++){
        // iter on each sequence
        for(const auto &tokens_ : tokens){
            get_bytes_pair_counts(tokens_, pair_counts);
        }

        // get the top frequent pair
        get_top_pair(pair_counts, top_pair);
        // check top_pair frequency
        if(pair_counts[top_pair] < 2){
            break;
        }

        // merge the top pair
        merge_map.push_back(vector<int>{top_pair.first, top_pair.second, token_id});
        for(auto &tokens_ : tokens){
            merge_bytes_pair(tokens_, top_pair, token_id);
        }

        // clear pair_counts
        pair_counts.clear();
        token_id++;
    }

    return merge_map;
}
