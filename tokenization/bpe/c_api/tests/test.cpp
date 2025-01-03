#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <omp.h>
#include "../core/include/base.h"
#include "../core/include/basic.h"

using namespace std;

string load_text_file(const string& file_path){
    // open file
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        printf("Cannot open file: %s\n", file_path.c_str());
        return "";
    }

    // read file
    std::string text((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
    file.close();
    return text;
}

vector<int> encode_utf8(string text){
    int size_ = text.size();
    vector<int> byte_list(size_);
    
    #pragma omp parallel for
    for(int i = 0; i < size_; i++){
        byte_list[i] = static_cast<unsigned char>(text[i]);
    }
    return byte_list;
}

void test_taylor_swift_text(int vocab_size=500){
    // load text file and encode with utf-8
    vector<vector<int>> tokens;
    string text = load_text_file("../../TaylorSwift.txt");
    tokens.push_back(encode_utf8(text));
    printf("encode utf-8 done, tokens size: %ld\n", tokens[0].size());

    // count run time
    auto start = chrono::high_resolution_clock::now();
    auto merge_map = bytes_pair_encoding(tokens, vocab_size);
    auto end = chrono::high_resolution_clock::now();

    // print run time in seconds
    double duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    printf("Time: %.4f seconds\n", duration / 1000);

    // print the last 100 pairs
    for (int i = 4700; i < merge_map.size(); i++){
        printf("Pair: %d %d %d\n", merge_map[i][0], merge_map[i][1], merge_map[i][2]);
    }
    // for (int i = 0; i < merge_map.size(); i++){
    //     printf("Pair: %d %d %d\n", merge_map[i][0], merge_map[i][1], merge_map[i][2]);
    // }
}

int main(){
    // test_get_bytes_pair_counts();
    // test_merge_bytes_pair();
    test_taylor_swift_text(5000);
    return 0;
}