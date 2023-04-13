#pragma once
#include <memory>

template <class T>
class CircularBuffer {
public:
	T * buffer;
	//tail points to the last added element
	size_t size, head, tail;
	bool full;
public:
	CircularBuffer(size_t size) {
		this->buffer = new T[size]; //size no debe estar en bytes sino en la cantidad de elementos de tipo T del buffer
		memset(this->buffer, 0, size*sizeof(T));
		this->size = size;
		this->head = 0;
		this->tail = 0;
		this->full = false;
	}
	void insert(T * source, size_t size) {
		//First check if insert has to be done in two steps
		if (this->tail + size >= this->size) {
			size_t a = sizeof(T);
			size_t first_cpy_size = this->size - 1 - this->tail;
			memcpy(&this->buffer[this->tail + 1], source, first_cpy_size*sizeof(T));
			memcpy(this->buffer, &source[first_cpy_size], (size - first_cpy_size)*sizeof(T));
			this->tail = size - first_cpy_size - 1;
			this->head = (this->tail + 1) % this->size;
			this->full = true;
		}
		else { 
			if (tail == head) {
				memcpy(&this->buffer[this->tail], source, size * sizeof(T));
				this->tail = this->tail + size - 1;
			}
			else {
				memcpy(&this->buffer[this->tail + 1], source, size * sizeof(T));
				this->tail = this->tail + size;
			}
			//If buffer was full we need to move the head
			if (this->full) {
				this->head = (this->tail + 1) % this->size;
			}
			if (this->tail == this->size - 1) {
				full = true;
			}
		}
	}
	//get the position of the element in absolute array index. Begginig at 1 - buffer[0] is the first element.
	size_t getElementPos(size_t iter) {
		size_t ret = 0;
		if (head > iter) {
			ret = iter + 1 + (this->size - this->head);
		}
		else {
			ret = iter + 1 - this->head;
		}
		return ret;
	}

	//get the element with absolute position pos
	T getElement(size_t pos) {
		if (head + pos < size) {
			return buffer[head + pos];
		}
		else {
			size_t remainder = pos - (size - head);
			return buffer[remainder];
		}
	}

	//set the element with absolute position pos
	void setElement(size_t pos, T elem) {
		if (head + pos < size) {
			buffer[head + pos] = elem;
		}
		else {
			size_t remainder = pos - (size - head);
			buffer[remainder] = elem;
		}
	}

	void copyElements(T * output, size_t size) {
		if (this->tail < size - 1) {
			size_t carry_over = (size - 1) - this->tail;
			memcpy(output, &this->buffer[this->size - carry_over], carry_over * sizeof(T));
			memcpy(&output[carry_over], this->buffer, (this->tail + 1) * sizeof(T));
		}
		else {
			memcpy(output, &this->buffer[this->tail - (size - 1)], size * sizeof(T));
			//Quizas deberia tener ambos valores: nframes y nbytes para no tener que calcular ninguno
		}
	}
	//void addToOutput(T * output, size_t sampleIdx, double propagationTime, float remainingEnergyFactor) {
	//	if (time_of_listen < renderData->samplesRecordBuffer->size &&
	//		time_of_listen >= renderData->samplesRecordBuffer->size - renderData->bufferFrames) {
	//		//Need to convert time_of_listen to output index
	//		renderData->samplesRecordBuffer->addToOutput(index, renderData->paths->ptr[i].remaining_energy_factor * renderData->samplesRecordBuffer->buffer[iter]);
	//		renderData->samplesRecordBuffer->addToOutput(index + 1, renderData->paths->ptr[i].remaining_energy_factor * renderData->samplesRecordBuffer->buffer[iter + 1]);
	//	}
	//}

	void insertSampleElements(T * source, size_t start, size_t size) {
		memcpy(&this->buffer[start], source, size * sizeof(T));
	}

	T& operator[](size_t idx){
		return this->buffer[idx];
	}
	~CircularBuffer() {
		delete[](this->buffer);
	}
};

template <class T>
class SampleIterator {
public:
	CircularBuffer<T> * cb;
	int numChannels;
	size_t idx;
	bool finished;
public:
	SampleIterator(CircularBuffer<T> * cb, int numChannels) {
		this->cb = cb;
		this->numChannels = numChannels;
		this->idx = cb->head;
		this->finished = false;
	}
	void next() {
		this->idx = (idx + numChannels) % cb->size;
		if (idx == cb->tail-(numChannels-1)) { //in case of 2 channels the second to last sample is the first of the last pair 
			this->finished = true;
		}
	}
	bool isEnd() {
		return this->finished;
	}
};