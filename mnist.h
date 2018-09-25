#pragma once

#include "util.h"

#include <string>

namespace boostnet {

matrix load_mnist_images(const std::string& path);
matrix load_mnist_labels(const std::string& path);

}
