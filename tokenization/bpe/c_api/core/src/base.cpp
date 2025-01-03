// base.cpp
#include "../include/base.h"
#include <iostream>

using namespace std;

/*
get the pair counts of byte_list

@param byte_list: list of bytes
@param pair_counts: pair counts of byte_list
*/ 
void get_bytes_pair_counts(
    const vector<int>& byte_list,
    unordered_map<pair<int, int>, int, pair_hash> &pair_counts
){
    int size_ = byte_list.size();
    if(size_ < 2) return;

    for(int i = 0; i < size_ - 1; i++){
        pair<int, int> pair_ = make_pair(byte_list[i], byte_list[i + 1]);
        if(pair_counts.find(pair_) == pair_counts.end()){
            pair_counts[pair_] = 1;
        }else{
            pair_counts[pair_] += 1;
        }
    }
}


/*
merge the byte pair in the byte_list with the given pair in-place

@param byte_list: list of bytes
@param pair_: pair to merge
@param token_id: new token id for the merged pair
*/
void merge_bytes_pair(
    vector<int>& byte_list,
    const pair<int, int>& pair_,
    const int token_id
){
    // init merged byte list
    int valid_len = byte_list.size();
    int size_tail = valid_len - 1;
    int i = 0;
    int j = 0;

    while(i < valid_len){
        if(i < size_tail && byte_list[i] == pair_.first && byte_list[i + 1] == pair_.second){
            byte_list[j] = token_id;
            i += 2;
        }else{
            byte_list[j] = byte_list[i];
            i += 1;
        }
        j += 1;
    }

    // valid_len = j;
    byte_list.resize(j);
}


/*
get the top pair from pair_counts and store it in top_pair
*/
void get_top_pair(
    const unordered_map<pair<int, int>, int, pair_hash>& pair_counts,
    pair<int, int>& top_pair
){
    // create a max function using pair_counts' values
    auto max_pair = max_element(
        pair_counts.begin(),
        pair_counts.end(),
        []( const pair<pair<int, int>, int> &p1,
            const pair<pair<int, int>, int> &p2){
            return p1.second < p2.second;
        }
    );

    // get the top pair
    top_pair = max_pair->first;
}