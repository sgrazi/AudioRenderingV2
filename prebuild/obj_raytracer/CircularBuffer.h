#ifndef CIRCULARBUFFER_H
#define CIRCULARBUFFER_H

#include <vector>
#include <stdexcept>
#include <iostream>

template <typename T>
class CircularBuffer {
public:
    CircularBuffer(size_t size) : size(size), buffer(size, 0), index(0) {}

    void add(const T* values, size_t length) {
        size_t initial_index = index;
        for (size_t i = 0; i < length; ++i) {
            buffer[index] += values[i];
            index = (index + 1) % size;
        }
        index = initial_index;
    }


    std::vector<T> get_and_reset(size_t n) {
        if (n > size) {
            throw std::invalid_argument("Requested more elements than present in the buffer");
        }

        std::vector<T> result;
        for (int i = 0; i < n; i++) {
            T local_value = buffer[(index + i) % size];
            buffer[(index + i) % size] = 0.0;
            result.push_back(local_value);
        }
        index = (index + n) % size;
        return result;
    }

    void print_buffer() const {
        for (const auto& value : buffer) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }


private:
    size_t size;
    std::vector<T> buffer;
    size_t index;
};

#endif // CIRCULARBUFFER_H