#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>

int main() {
    std::ifstream in("graph.bin", std::ios::binary | std::ios::ate);
    if (!in) { std::cerr << "graph.bin not found\n"; return 1; }

    std::streamsize file_size = in.tellg();      // bytes on disk
    in.seekg(0, std::ios::beg);

    uint32_t n;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));

    std::uint64_t expected = 4ull + std::uint64_t(n) * n;
    if (file_size != static_cast<std::streamsize>(expected)) {
        std::cerr << "Size mismatch: header says N = " << n
                  << " ⇒ expected " << expected
                  << " bytes, but file is " << file_size << " bytes.\n";
        return 1;
    }

    std::vector<uint8_t> buf(n * n);
    in.read(reinterpret_cast<char*>(buf.data()), buf.size());

    if (!in) { std::cerr << "Unexpected EOF while reading matrix\n"; return 1; }

    // Success ─ print matrix
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            std::cout << int(buf[i*n + j]) << (j + 1 == n ? '\n' : ' ');
        }
    }
}
