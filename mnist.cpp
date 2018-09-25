#include "mnist.h"

#include <vector>
#include <fstream>
#include <stdexcept>
#include <inttypes.h>

namespace boostnet {

static std::vector<char> read_file(const std::string& path) {
    std::ifstream file(path);

    return std::vector<char>(
                (std::istreambuf_iterator<char>(file)),
                std::istreambuf_iterator<char>());
}

static uint32_t reverse_bytes(uint32_t v) {
    return ((v & 0xFF) << 24) | ((v & 0xFF00) << 8) | ((v & 0xFF0000) >> 8) | ((v & 0xFF000000) >> 24);
}

#pragma pack(push, 1)

struct mnist_images_header {
    uint32_t magic;
    uint32_t image_count;
    uint32_t row_count;
    uint32_t col_count;
    unsigned char data[];

    void post_load() {
        magic = reverse_bytes(magic);
        image_count = reverse_bytes(image_count);
        row_count = reverse_bytes(row_count);
        col_count = reverse_bytes(col_count);
    }
};

struct mnist_labels_header {
    uint32_t magic;
    uint32_t label_count;
    unsigned char data[];

    void post_load() {
        magic = reverse_bytes(magic);
        label_count = reverse_bytes(label_count);
    }
};

#pragma pack(pop)

matrix load_mnist_images(const std::string& path) {
    std::vector<char> data = read_file(path);
    mnist_images_header* const images =
            reinterpret_cast<mnist_images_header*>(data.data());
    images->post_load();

    if (images->magic != 2051) {
        throw std::logic_error("bad file");
    }

    matrix res(images->image_count, images->row_count * images->col_count);

    size_t index = 0;
    for (size_t i = 0; i < images->image_count; ++i) {
        for (size_t j = 0; j < images->row_count; ++j) {
            for (size_t k = 0; k < images->col_count; ++k) {
                res(i, j * images->col_count + k) = images->data[index++] / 255.0f;
            }
        }
    }

    return res;
}

matrix load_mnist_labels(const std::string& path) {
    std::vector<char> data = read_file(path);
    mnist_labels_header* const labels =
            reinterpret_cast<mnist_labels_header*>(data.data());
    labels->post_load();

    if (labels->magic != 2049) {
        throw std::logic_error("bad file");
    }

    size_t max_label = 0;
    for (size_t i = 0; i < labels->label_count; ++i) {
        if (labels->data[i] > max_label) {
            max_label = labels->data[i];
        }
    }

    matrix res;
    res.setZero(labels->label_count, max_label + 1);

    for (size_t i = 0; i < labels->label_count; ++i) {
        res(i, labels->data[i]) = 1;
    }

    return res;
}

}
