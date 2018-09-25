#include "functions.h"
#include "mnist.h"
#include "cifar.h"
#include "model.h"
#include "optimize.h"
#include "random.h"

#include <iostream>
#include <fstream>

#include <fenv.h>

template<typename T>
void save_matrix(const T& matrix, const std::string& file_name) {
    static Eigen::IOFormat format(
                Eigen::FullPrecision,
                Eigen::DontAlignCols,
                "\t");

    std::ofstream file(file_name);

    file << matrix.format(format);
}

boostnet::matrix load_matrix(const std::string& file_name) {
    std::ifstream file(file_name);

    size_t rows, cols;
    file >> rows >> cols;

    boostnet::matrix res(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            boostnet::scalar_type value;
            file >> value;

            res(i, j) = value;
        }
    }

    return res;
}

void tree_test2() {
    std::vector<uint32_t>              features = {0, 1};
    std::vector<boostnet::scalar_type> thresholds = {0, 0};
    std::vector<boostnet::row_vector>  values(4);
    values[0] = boostnet::row_vector::Zero(1, 1);
    values[1] = boostnet::row_vector::Ones(1, 1);
    values[2] = boostnet::row_vector::Ones(1, 1);
    values[3] = boostnet::row_vector::Zero(1, 1);

    boostnet::oblivious_tree t;//(features, thresholds, values, {0});

    int xi = 0;
    for (float i = -3; i <= 3; i += 0.1) {
        int xj = 0;
        for (float j = -3; j <= 3; j += 0.1) {
            boostnet::matrix x(1, 2);
            x << i, j;

            boostnet::matrix g = boostnet::matrix::Zero(1, 2);
            t.gradient(x, boostnet::matrix::Ones(1, 1), g);

            boostnet::matrix f = boostnet::matrix::Zero(1, 2);
            t.call(x, f);

            std::cout << xi << '\t' << xj << '\t' <<
                         i << '\t' << j << '\t' <<
                         g(0, 0) << '\t' << g(0, 1) << '\t' <<
                         f(0, 0) << '\t' << f(0, 1) << '\t' <<
                         std::endl;

            ++xj;
        }
        ++xi;
    }
}

void tree_test3() {
    boostnet::column_vector x(401);
    for (int i = -200; i <= 200; ++i) {
        x(i + 200) = i / 100.0;
    }

    boostnet::column_vector y = -x;
    y = (1.0f / (1.0f + Eigen::exp(y.array()))).matrix();

    boostnet::oblivious_tree t;
    t.grow(10, true, x, y, {0}, {0});

    return;
}

void forest_test() {
    boostnet::matrix x1 = boostnet::fill_normal<boostnet::matrix>(2000, 2);
    boostnet::matrix x2 = boostnet::fill_normal<boostnet::matrix>(2000, 2, 2);
    boostnet::matrix x3 = boostnet::fill_normal<boostnet::matrix>(2000, 2, -2);

    boostnet::matrix X(x1.rows() + x2.rows() + x3.rows(), x1.cols());
    X << x1, x2, x3;

    boostnet::matrix Y(X.rows(), 2);
    for (int i = 0; i < Y.rows(); ++i) {
        if (i < X.rows() / 3) {
            Y(i, 0) = 1;
            Y(i, 1) = 0;
        } else {
            Y(i, 0) = 0;
            Y(i, 1) = 1;
        }
    }

    const auto x = boostnet::make_constant(X.cols());
    const auto y = boostnet::make_constant(Y.cols());
    const auto forest = boostnet::make_forest<boostnet::oblivious_tree>(
                x, 2,
                boostnet::forest_config()
                    .set_leaf_regularization(0)
                    .set_tree_depth(5)
                    .set_min_trees(0)
                    .set_max_trees(100000)
                    .set_reuse_features(false)
                    .set_dropout(0));
    const auto prob = boostnet::make_sigmoid(forest);
    const auto mse = boostnet::make_mse(prob, y);

    boostnet::model model(mse);

    boostnet::sgd(
                model,
                {{x, X}, {y, Y}},
                {{x, X}, {y, Y}},
                1000,
                100,
                100);

    save_matrix(X, "/home/kmosienko/X.tsv");
    save_matrix(Y, "/home/kmosienko/Y.tsv");

    model.set_input(x, X).set_input(y, Y);
    model.forward();

    save_matrix(model.get_output(prob), "/home/kmosienko/prob.tsv");
    save_matrix(model.get_output(forest), "/home/kmosienko/forest.tsv");

    std::ofstream out("/home/kmosienko/forest2.tsv");
    boostnet::model forest_model(forest);
    int xi = 0;
    for (float i = -6; i <= 6; i += 1) {
        int xj = 0;
        for (float j = -6; j <= 6; j += 1) {
            boostnet::matrix X(1, 2);
            X << i, j;

            forest_model.set_input(x, X);
            forest_model.forward();
            auto start_g = boostnet::matrix(1, 2);

            const auto dist_to_zero = X.array().square().sum();
            const auto dist_to_two = (X.array() - 2).square().sum();
            const auto dist_to_mtwo = (X.array() + 2).square().sum();

            if (dist_to_zero < dist_to_two && dist_to_zero < dist_to_mtwo) {
                start_g << 1, 0;
            } else {
                start_g << 0, 1;
            }
            forest_model.backward(start_g, false);

            boostnet::matrix g = forest_model.get_output_gradient(x);

            boostnet::matrix f = forest_model.get_output(forest);

            out << xi << '\t' << xj << '\t' <<
                         i << '\t' << j << '\t' <<
                         g(0, 0) << '\t' << g(0, 1) << '\t' <<
                         f(0, 0) << '\t' << f(0, 1) << '\t' <<
                         std::endl;

            ++xj;
        }
        ++xi;
    }
}

void forest_test2() {
    boostnet::matrix x1 = boostnet::fill_normal<boostnet::matrix>(2000, 2, -2);
    boostnet::matrix x2 = boostnet::fill_normal<boostnet::matrix>(2000, 2, 0);
    boostnet::matrix x3 = boostnet::fill_normal<boostnet::matrix>(2000, 2, 2);

    boostnet::matrix X(x1.rows() + x2.rows() + x3.rows(), x1.cols());
    X << x1, x2, x3;

    boostnet::matrix Y(X.rows(), 3);
    for (int i = 0; i < Y.rows(); ++i) {
        if (i < X.rows() / 3) {
            Y(i, 0) = 1;
            Y(i, 1) = 0;
            Y(i, 2) = 0;
        } else if (i < (2 * X.rows()) / 3) {
            Y(i, 0) = 0;
            Y(i, 1) = 1;
            Y(i, 2) = 0;
        } else {
            Y(i, 0) = 0;
            Y(i, 1) = 0;
            Y(i, 2) = 1;
        }
    }

    const auto x = boostnet::make_constant(X.cols());
    const auto y = boostnet::make_constant(Y.cols());
    const auto forest = boostnet::make_forest<boostnet::oblivious_tree>(
                x, 3,
                boostnet::forest_config()
                    .set_leaf_regularization(0)
                    .set_tree_depth(5)
                    .set_min_trees(1000)
                    .set_max_trees(1000)
                    .set_reuse_features(false)
                    .set_dropout(0.2));
    const auto prob = boostnet::make_sigmoid(forest);
    const auto mse = boostnet::make_mse(prob, y);

    boostnet::model model(mse);

    boostnet::sgd(
                model,
                {{x, X}, {y, Y}},
                {{x, X}, {y, Y}},
                100,
                50,
                0.005);

    save_matrix(X, "/home/kmosienko/X.tsv");
    save_matrix(Y, "/home/kmosienko/Y.tsv");

    model.set_input(x, X).set_input(y, Y);
    model.forward();

    save_matrix(model.get_output(prob), "/home/kmosienko/prob.tsv");
    save_matrix(model.get_output(forest), "/home/kmosienko/forest.tsv");

    std::ofstream out("/home/kmosienko/forest2.tsv");
    boostnet::model forest_model(forest);
    int xi = 0;
    for (float i = -6; i <= 6; i += 1) {
        int xj = 0;
        for (float j = -6; j <= 6; j += 1) {
            boostnet::matrix X(1, 2);
            X << i, j;

            forest_model.set_input(x, X);
            forest_model.forward();
            auto start_g = boostnet::matrix(1, 3);

            start_g << 0, 0, 1;

            forest_model.backward(start_g, false);

            boostnet::matrix g = forest_model.get_output_gradient(x);

            boostnet::matrix f = forest_model.get_output(forest);

            out << xi << '\t' << xj << '\t' <<
                         i << '\t' << j << '\t' <<
                         g(0, 0) << '\t' << g(0, 1) << '\t' << g(0, 1) << '\t' <<
                         f(0, 0) << '\t' << f(0, 1) << '\t' << f(0, 2) << '\t' <<
                         std::endl;

            ++xj;
        }
        ++xi;
    }
}

void mnist_forest_classification() {
    boostnet::matrix images_train = boostnet::load_mnist_images("/home/kmosienko/mnist/train-images-idx3-ubyte");
    boostnet::matrix labels_train = boostnet::load_mnist_labels("/home/kmosienko/mnist/train-labels-idx1-ubyte");

    boostnet::matrix images_test = boostnet::load_mnist_images("/home/kmosienko/mnist/t10k-images-idx3-ubyte");
    boostnet::matrix labels_test = boostnet::load_mnist_labels("/home/kmosienko/mnist/t10k-labels-idx1-ubyte");

    const auto x = boostnet::make_constant(images_train.cols());
    const auto y = boostnet::make_constant(labels_train.cols());

    const auto mse = boostnet::make_mse(
                boostnet::make_sigmoid(
                    boostnet::make_forest<boostnet::oblivious_tree>(x, labels_train.cols())),
                y);

    boostnet::model model(mse);

    boostnet::sgd(
                model,
                {{x, images_train}, {y, labels_train}},
                {{x, images_test}, {y, labels_test}},
                1000,
                1000,
                .1);
}

void mnist_dense_classification() {
    boostnet::matrix images_train = boostnet::load_mnist_images("/home/kmosienko/mnist/train-images-idx3-ubyte");
    boostnet::matrix labels_train = boostnet::load_mnist_labels("/home/kmosienko/mnist/train-labels-idx1-ubyte");

    boostnet::matrix images_test = boostnet::load_mnist_images("/home/kmosienko/mnist/t10k-images-idx3-ubyte");
    boostnet::matrix labels_test = boostnet::load_mnist_labels("/home/kmosienko/mnist/t10k-labels-idx1-ubyte");

    const auto x = boostnet::make_constant(images_train.cols());
    const auto y = boostnet::make_constant(labels_train.cols());

    const auto l1 =
            boostnet::make_sigmoid(
                boostnet::make_linear(x, 100));

    const auto l2 =
            boostnet::make_sigmoid(
                boostnet::make_linear(l1, 50));

    const auto mse = boostnet::make_mse(
                boostnet::make_sigmoid(
                    boostnet::make_linear(l2, labels_train.cols())),
                y);

    boostnet::model model(mse);

    boostnet::sgd(
                model,
                {{x, images_train}, {y, labels_train}},
                {{x, images_test}, {y, labels_test}},
                10,
                1000,
                1);
}

void mnist_dense_autoencoder() {
    //boostnet::matrix images_train = boostnet::load_mnist_images("/home/kmosienko/mnist/train-images-idx3-ubyte");
    //boostnet::matrix images_test = boostnet::load_mnist_images("/home/kmosienko/mnist/t10k-images-idx3-ubyte");

    boostnet::matrix images_train = load_matrix("/home/kmosienko/mnist/mnist8x8_images.txt") / 16;
    boostnet::matrix images_test = images_train;

    const float dropout = 0.0;

    const auto x = boostnet::make_constant(images_train.cols());

    const auto encoder =
            boostnet::make_sigmoid(
                boostnet::make_linear(
                    boostnet::make_dropout(x, dropout), 10));

    const auto decoder =
            boostnet::make_sigmoid(
                boostnet::make_linear(
                    boostnet::make_dropout(encoder, dropout), images_train.cols()));

    boostnet::model model(boostnet::make_mse(decoder, x));

    boostnet::sgd(
                model,
                {{x, images_train}},
                {{x, images_test}},
                50,
                1000,
                .001,
                "/home/kmosienko/dense_ae.log");

    model.set_input(x, images_test);
    model.forward();

    //save_matrix(boostnet::load_mnist_labels("/home/kmosienko/mnist/t10k-labels-idx1-ubyte"), "/home/kmosienko/labels_test.tsv");
    save_matrix(load_matrix("/home/kmosienko/mnist/mnist8x8_labels.txt"), "/home/kmosienko/labels_test.tsv");
    save_matrix(model.get_output(encoder), "/home/kmosienko/dense_encoder.tsv");
    save_matrix(model.get_output(decoder), "/home/kmosienko/dense_decoder.tsv");
}

void mnist_forest_autoencoder() {
    boostnet::matrix images_train = boostnet::load_mnist_images("/home/kmosienko/mnist/train-images-idx3-ubyte");
    boostnet::matrix images_test = boostnet::load_mnist_images("/home/kmosienko/mnist/t10k-images-idx3-ubyte");

    //boostnet::matrix images_train = load_matrix("/home/kmosienko/mnist/mnist8x8_images.txt") / 16;
    //boostnet::matrix images_test = images_train;

    //boostnet::matrix images_train = boostnet::load_cifar("/home/kmosienko/cifar/data_batch.bin");
    //boostnet::matrix images_test = boostnet::load_cifar("/home/kmosienko/cifar/test_batch.bin");

    boostnet::forest_config config;
    config.tree_depth = 5;
    config.min_trees = 300;
    config.max_trees = 300;
    config.x_subsample = 0.2;
    config.y_subsample = 0.2;
    config.dropout = 0.0;

    const auto x = boostnet::make_constant(images_train.cols());
    const auto encoder = boostnet::make_forest<boostnet::oblivious_tree>(
                x, 20,
                config);
//    const auto decoder = boostnet::make_forest<boostnet::oblivious_tree>(
//                encoder, images_train.cols(),
//                config);
    const auto decoder =
            boostnet::make_sigmoid(
                boostnet::make_forest<boostnet::oblivious_tree>(
                    encoder, images_train.cols(),
                    config));

    boostnet::model model(boostnet::make_mse(decoder, x));

    boostnet::sgd(
                model,
                {{x, images_train}},
                {{x, images_test}},
                //images_train.rows(),
                100,
                5,
                .0001,
                "/home/kmosienko/test_1.log");

    model.set_input(x, images_test);
    model.forward();

    save_matrix(boostnet::load_mnist_labels("/home/kmosienko/mnist/t10k-labels-idx1-ubyte"), "/home/kmosienko/labels_test.tsv");
    //save_matrix(load_matrix("/home/kmosienko/mnist/mnist8x8_labels.txt"), "/home/kmosienko/labels_test.tsv");
    save_matrix(model.get_output(encoder), "/home/kmosienko/encoder.tsv");
    save_matrix(model.get_output(decoder), "/home/kmosienko/decoder.tsv");

    save_matrix(images_test, "/home/kmosienko/images_test.tsv");

//    model.backward();
//    save_matrix(-model.get_output_gradient(encoder), "/home/kmosienko/encoder_grad.tsv");
}

int main(int, char**) {
    std::ios_base::sync_with_stdio(0);

//    save_matrix(load_matrix("/home/kmosienko/mnist/mnist8x8_images.txt") / 16,
//                "/home/kmosienko/8x8_truth.tsv");

//    return 0;

    //tree_test();
    //tree_test2();
    //tree_test3();
    //sigmoid_test();
    //forest_test();
    //forest_test2();
    //mnist_forest_classification();
    //mnist_dense_classification();
    mnist_dense_autoencoder();
    //mnist_forest_autoencoder();

    return 0;
}
