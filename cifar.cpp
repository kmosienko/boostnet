#include "cifar.h"
#include "util.h"

#include <vector>
#include <fstream>
#include <stdexcept>
#include <inttypes.h>

namespace boostnet {

template<typename C>
static std::vector<C> read_file(const std::string& path) {
    std::ifstream file(path);

    return std::vector<C>(
                (std::istreambuf_iterator<char>(file)),
                std::istreambuf_iterator<char>());
}

matrix load_cifar(const std::string& path) {
    std::vector<unsigned char> data = read_file<unsigned char>(path);

    BN_VERIFY(data.size() % 3073 == 0);
    const size_t image_count = data.size() / 3073;

    matrix res(image_count, 32 * 32);

    for (size_t i = 0; i < image_count; ++i) {
        const unsigned char* const image = &data[i * 3073 + 1];
        for (size_t j = 0; j < 32; ++j) {
            for (size_t k = 0; k < 32; ++k) {
                const scalar_type r = image[j * 32 + k + 0] / 255.0;
                const scalar_type g = image[j * 32 + k + 1024] / 255.0;
                const scalar_type b = image[j * 32 + k + 2048] / 255.0;

                const scalar_type pixel = (r + g + b) / 3.0;

                res(i, j * 32 + k) = 1.0 - pixel;
            }
        }
    }

    return res;
}

}
